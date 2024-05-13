import torch
import torch.nn as nn
import torch.nn.functional as F 
from transformers import LlamaForCausalLM,LlamaTokenizer
from tqdm import trange
import numpy as np
import json


class Sample_from_openllama_dataset(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.model_name=args.model_name
        self.sample_size=args.sample_size
        self.task_name=args.task_name
        self.token_num=args.token_num
        self.batch_size=args.batch_size
        
        self.model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b")
        self.tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b")
        self.layers=self.model.model.layers
        self.device=args.device
            
    @property
    def device(self):
        return self.model.device

    @device.setter
    def device(self, device):
        print(f'Model: set device to {device}')
        self.model = self.model.to(device)
        self.layers = self.layers.to(device)


        
    
            
    def forward(self,past_key_values = None,cache_position= None,position_ids = None,):
        # inputs = self.tokenizer("Q: What is the largest animal?\nA", return_tensors="pt").to(self.device)
        # outputs =self.model(**inputs, labels=inputs["input_ids"])
        text=self.get_text(self.token_num)
        inputs = self.tokenizer(text, return_tensors="pt")
        assert inputs["attention_mask"].size()[-1]==self.token_num
        attention_mask=inputs["attention_mask"].to(self.device)
        #print(attention_mask)
        attention_mask=attention_mask.expand(self.batch_size,-1)
        batch_size=attention_mask.size()[0]
        assert batch_size==self.batch_size
        
        head_mask = head_mask = [None] * 12
        for s in trange(self.sample_size):
            batch_data=self.random_select(self.batch_size,s)
            inputs_token = self.tokenizer(batch_data, return_tensors="pt").to(self.device)
            i=0
            inputs_embeds=self.model.model.embed_tokens(inputs_token['input_ids'])
            for decoder_layer in self.layers:
                
                # x_sample = self.gen_x(batch_size,self.token_num, 3200, 0.81)
                # inputs_embeds = torch.tensor(x_sample,dtype=torch.float32).to(self.device)
                #print(inputs_embeds)
                #print(hidden_states.cpu())
                if i==25: 
                    input_save_dir='/sample_'+str(s)+'input_layer'+str(i)+"_token"+str(self.token_num)+'_batch'+str(self.batch_size)+'.pth'
                    torch.save(inputs_embeds[:,:,:768].cpu(),self.args.save_folder+input_save_dir)
                past_seen_tokens = 0
                if cache_position is None:
                    cache_position = torch.arange(
                        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                    )
                if position_ids is None:
                    position_ids = cache_position.unsqueeze(0)
                causal_mask = self.model.model._update_causal_mask(attention_mask, inputs_embeds, cache_position)
                hidden_states = inputs_embeds
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=False,
                    use_cache=True,
                    cache_position=cache_position,
                )
                inputs_embeds = layer_outputs[0].detach()
                if i==25:
                    output_save_dir='/sample_'+str(s)+'output_layer'+str(i)+"_token"+str(self.token_num)+'_batch'+str(self.batch_size)+'.pth'
                    torch.save(hidden_states[:,:,:768].cpu(),self.args.save_folder+output_save_dir)
                i=i+1
        return 1
    
    def get_text(self,num):
        text='A,'
        for i in range(int(num-(0.5*num)-1)):
            text=text+str(i)+','
        return text
    
    def gen_x(self,num, token_num,dim ,power):
        return np.random.normal(0., np.sqrt(power), [num,token_num, dim])
    
    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None
    
    
    def random_select(self,batch_size,sample_num):
        with open('data/MIdataset/all_color_sup_token.json','r') as file:
            all_data=json.load(file)

        batch_data=all_data[sample_num*batch_size:(sample_num+1)*batch_size]
        file.close()
        return batch_data


class Sample_with_context(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.model_name=args.model_name
        self.sample_size=args.sample_size
        self.task_name=args.task_name
        self.token_num=args.token_num
        self.batch_size=args.batch_size
        
        self.model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b")
        self.tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b")
        self.layers=self.model.model.layers
        self.device=args.device
            
    @property
    def device(self):
        return self.model.device

    @device.setter
    def device(self, device):
        print(f'Model: set device to {device}')
        self.model = self.model.to(device)
        self.layers = self.layers.to(device)

            

        
    
            
    def forward(self,past_key_values = None,cache_position= None,position_ids = None,):
        
        
        # inputs_token = self.tokenizer('the', return_tensors="pt").to(self.device)
        
        # outputs =self.model(**inputs, labels=inputs["input_ids"])
        text=self.get_text(self.token_num)
        #text='I,'
        inputs = self.tokenizer(text, return_tensors="pt")
        print(inputs["attention_mask"].size()[-1])
        assert inputs["attention_mask"].size()[-1]==self.token_num
        attention_mask=inputs["attention_mask"].to(self.device)
        #print(attention_mask)
        attention_mask=attention_mask.expand(self.batch_size,-1)
        batch_size=attention_mask.size()[0]
        assert batch_size==self.batch_size
        
        head_mask = head_mask = [None] * 12
        for s in trange(self.sample_size):
            
           
                
            i=0    
            #inputs = self.tokenizer("Hello, my dog is cute", return_tensors="pt")
            batch_input_token=torch.load('data/MIdataset/context_tensor/{}_batch500_OpenHermes.pth'.format(str(s))).to(self.device)
            batch_input_token=batch_input_token[:,16-self.token_num:]
            inputs_embeds=self.model.model.embed_tokens(batch_input_token)
            for decoder_layer in self.layers:
                
                # x_sample = self.gen_x(batch_size,self.token_num, 3200, 0.81)
                # inputs_embeds = torch.tensor(x_sample,dtype=torch.float32).to(self.device)
                #print(inputs_embeds)
                #print(hidden_states.cpu())
                if i==20 or i==24 or i==25:  
                    input_save_dir='/sample_'+str(s)+'input_layer'+str(i)+"_token"+str(self.token_num)+'_batch'+str(self.batch_size)+'.pth'
                    torch.save(inputs_embeds[:,:,:768].cpu(),self.args.save_folder+input_save_dir)
                past_seen_tokens = 0
                if cache_position is None:
                    cache_position = torch.arange(
                        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                    )
                if position_ids is None:
                    position_ids = cache_position.unsqueeze(0)
                causal_mask = self.model.model._update_causal_mask(attention_mask, inputs_embeds, cache_position)
                hidden_states = inputs_embeds
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=False,
                    use_cache=True,
                    cache_position=cache_position,
                )
                inputs_embeds = layer_outputs[0].detach()
                if i==20 or i==24 or i==25:  
                    output_save_dir='/sample_'+str(s)+'output_layer'+str(i)+"_token"+str(self.token_num)+'_batch'+str(self.batch_size)+'.pth'
                    torch.save(hidden_states[:,:,:768].cpu(),self.args.save_folder+output_save_dir)
                i=i+1
        return 1
    
    def get_text(self,num):
        text=''
        for i in range(num-2):
            text=text+str(i)
        return text
    
    def gen_x(self,num, token_num,dim ,power):
        return np.random.normal(0., np.sqrt(power), [num,token_num, dim])
        

    def random_select(self,batch_size,sample_num):
        with open('data/MIdataset/all_text_10token.json','r') as file:
            all_data=json.load(file)

        batch_data=all_data[sample_num*batch_size:(sample_num+1)*batch_size]
        file.close()
        return batch_data


