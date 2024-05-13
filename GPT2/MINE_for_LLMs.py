import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"]='2'
torch.set_num_threads(8)
cuda=torch.cuda.is_available()
print(cuda)
from transformers import get_linear_schedule_with_warmup,get_polynomial_decay_schedule_with_warmup
import csv

class Net(nn.Module):
    def __init__(self,x_dim,y_dim) :
        super((Net), self).__init__()
        self.layers = nn.Sequential(
            nn.Linear((x_dim+y_dim), 768),
            nn.LeakyReLU(),
            nn.Linear(768, 384),
            nn.LeakyReLU(),
            nn.Linear(384, 192),
            nn.LeakyReLU(),
            nn.Linear(192, 96),
            nn.LeakyReLU(),
            nn.Linear(96, 48),
            nn.LeakyReLU(),
            nn.Linear(48, 24),
            nn.LeakyReLU(),
            nn.Linear(24, 12),
            nn.LeakyReLU(),
            nn.Linear(12, 6),
            nn.LeakyReLU(),
            nn.Linear(6, 3),
            nn.LeakyReLU(),
            nn.Linear(3, 1))
        # self.layers = nn.Sequential(
        #     nn.Linear((x_dim+y_dim), 3072),
        #     nn.LeakyReLU(),
        #     nn.Linear(3072, 1536),
        #     nn.LeakyReLU(),
        #     nn.Linear(1536, 768),
        #     nn.LeakyReLU(),
        #     nn.Linear(768, 384),
        #     nn.LeakyReLU(),
        #     nn.Linear(384, 192),
        #     nn.LeakyReLU(),
        #     nn.Linear(192, 96),
        #     nn.LeakyReLU(),
        #     nn.Linear(96, 48),
        #     nn.LeakyReLU(),
        #     nn.Linear(48, 24),
        #     nn.LeakyReLU(),
        #     nn.Linear(24, 12),
        #     nn.LeakyReLU(),
        #     nn.Linear(12, 6),
        #     nn.LeakyReLU(),
        #     nn.Linear(6, 3),
        #     nn.LeakyReLU(),
        #     nn.Linear(3, 1))


    def forward(self, batch_size,inputs):
        
        logits = self.layers(inputs)#[2*batch,1]

        pred_xy = logits[:batch_size]#jointly distribution, Xsample and Ysample meet the same index (co-occurence) 
        pred_x_y = logits[batch_size:]#marginal distribution, Xsample and Ysample meet the different index (indepenent) 
        loss = - np.log2(np.exp(1)) * (torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y))))
        
        return loss
    

        

#估计器
class Estimator():
    def __init__(self,x_dim,y_dim) -> None:
        self.net = Net(x_dim,y_dim).cuda()
        self.net.apply(self.weight_init)
        #self.net.load_state_dict(torch.load('model_save/model_batch200000_dim10.pth'))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        #self.optimizer.load_state_dict(torch.load('model_save/optimizer_dim200000_epoch10.pth'))
        self.scheduler=get_polynomial_decay_schedule_with_warmup(self.optimizer, num_warmup_steps=5, num_training_steps=10000, lr_end=1e-8,power=3)
        #self.scheduler=get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=5, num_training_steps=10000)
        self.x_dim = x_dim
        self.y_dim = y_dim

    def backward(self,batch_size,input):
        loss = self.net(batch_size,input)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1, norm_type=2)
        self.net.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        info = -loss.detach()
        return info
    
    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

def main(layer,token,segment):  
    print('start to run layer {} & token {}'.format(layer,token))      
    n_epoch = 10000
    x_dim = 768
    y_dim = 768


    writer = SummaryWriter('./log')
    estimator = Estimator(x_dim,y_dim)

    path='data/sample_from_context/token_num8/'
    # layer=1
    # token=1
    x_sample_batch=None
    y_sample_batch=None
    for i in range(60):
        input_file='sample_'+str(i)+'input_layer'+str(layer)+'_token8_batch5000.pth'
        output_file='sample_'+str(i)+'output_layer'+str(layer)+'_token8_batch5000.pth'
        x_sample=torch.load(path+input_file).cuda()
        x_sample=x_sample.permute(1,2,0)
        y_sample=torch.load(path+output_file).cuda()
        y_sample=y_sample.permute(1,2,0)
        if x_sample_batch is None:
            x_sample_batch=x_sample[token][segment*x_dim:(segment+1)*x_dim]
            y_sample_batch=y_sample[token][segment*x_dim:(segment+1)*x_dim]
        else: 
            temp_x=x_sample[token][segment*x_dim:(segment+1)*x_dim]
            temp_y=y_sample[token][segment*x_dim:(segment+1)*x_dim]
            x_sample_batch=torch.cat((x_sample_batch,temp_x),dim=1)
            y_sample_batch=torch.cat((y_sample_batch,temp_y),dim=1)
    x_sample_batch=x_sample_batch.transpose(0,1).contiguous()
    y_sample_batch=y_sample_batch.transpose(0,1).contiguous()
    print('data is sucessfully loaded, the size is', x_sample_batch.size(), y_sample_batch.size())
    tiled_x = torch.cat([x_sample_batch, x_sample_batch, ], dim=0)#[2*batch,dim]
    batch_size = x_sample_batch.size(0)                



    info_max=0
    for epoch in tqdm(range(n_epoch)):

        idx = torch.randperm(batch_size).detach()#return a random list from 0 to batch_size-1 
        shuffled_y = y_sample_batch[idx]
        concat_y = torch.cat([y_sample_batch, shuffled_y], dim=0)#[2*batch,dim]
        inputs = torch.cat([tiled_x, concat_y], dim=1).detach()#[2*batch,2*dim]
        inputs=torch.tensor(inputs,dtype=torch.float32)
        info = estimator.backward(batch_size,inputs)
        if info_max<info and info==info:
            info_max=info
            
            
        elif info!=info:
            print('bad nan, jump to the next token')
            break
        
 

        writer.add_scalar('info',info,epoch)
    print('the {}-th layer and {}-th token has info_max: {}'.format(layer,token,info_max))
    return info_max.cpu().item()
    
    
if __name__ == '__main__':
    for i_layer in range(1,12):
        path='.csv'.format(i_layer)
        csvFile=open(path,"w+",newline='')
        name=['segment','token0','token1','token2','token3','token4','token5','token6','token7','token8','token9','token10','token11','token12','token13','token14','token15','token16','token17','token18','token19','token20']
        writer=csv.writer(csvFile)
        writer.writerow(name)
        csvFile.close()
    
    
    

        for i_segment in range(1):
            token_segment=[]
            token_segment.append(i_segment)
            for i_token in range(0,8):
                infor_max=main(layer=i_layer,token=i_token,segment=i_segment)
                token_segment.append(infor_max)
            with open(path,'a+') as f:
                csv_write=csv.writer(f)
                csv_write.writerow(token_segment)
                f.close()
            print("##########token {} has info: ".format(i_token), token_segment)
            
    