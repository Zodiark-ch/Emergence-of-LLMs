import sys,os,warnings,time,argparse,random
import logging
from sympy import im
warnings.filterwarnings("ignore")
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from representation_generator import simulate_micro_representation,simulate_macro_representation,true_mi
from dataloader import *
from MIestimatior_network import Estimator


parser = argparse.ArgumentParser()
parser.add_argument('--resource', default='test', type= str, help='dataset name, only dailydialog')
parser.add_argument('--gpu', type=str, default='0',  help='id of gpus')


parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--lr', type=float, default=6e-4, metavar='LR', help='learning rate')
parser.add_argument('--batch_size', type=int, default=10000, metavar='BS', help='batch size')
parser.add_argument('--epoch', type=int, default=4000, metavar='E', help='number of epochs')


parser.add_argument('--input_dimension', default='1000', type= int, help='the dimension of input representation')
parser.add_argument('--output_dimension', default='1000', type= int, help='the dimension of output representation')
parser.add_argument('--input_sample', default='10000', type= str, help='the number of sample of input representation')
parser.add_argument('--output_sample', default='10000', type= str, help='the number of sample of output representation')
parser.add_argument('--noise', default='0.2', type= str, help='bias')
parser.add_argument('--power', default='3', type= str, help='bias')
parser.add_argument('--variable_num', default='8', type= str, help='bias')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # logger.addHandler(sh)

    return logger

logger = get_logger('logs/MIestimation' + args.resource +'_'+str(args.lr)+'_'+str(args.epoch)+ '_logging.log')
logger.info('start training on GPU {}!'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
logger.info(args)

cuda=torch.cuda.is_available()

def macro_miestimator(path,input_dim,output_dim): 
    writer = SummaryWriter('./log')
    if args.resource=='test':
        macropath=simulate_macro_representation(args.output_sample,args.input_dimension,args.output_dimension,args.power,args.noise)
        true_mi_value=true_mi(args.power,args.noise,args.output_dimension)
        print('the datasets have been simulated, the true mutual information is {}'.format(true_mi_value))
    else: 
        macropath=path
        args.input_dimension=input_dim
        args.output_dimension=output_dim
    macro_data_loader=build_data(macropath,batch_size=args.batch_size)
    print('datasets have loaded') 
    
    estimator = Estimator(args.input_dimension,args.output_dimension)
    info_max=0
    early_stop=0
    for epoch in tqdm(range(args.epoch)):
        mi_mean_batch=0
        for train_step, batch in enumerate(macro_data_loader, 1):
            batch_input,batch_output=batch
            batch_input=batch_input.cuda()
            batch_output=batch_output.cuda()

            
            info = estimator.backward(batch_input,batch_output)
            #print(estimator.net.state_dict(), estimator.optimizer.state_dict())
            if info_max<info:
                info_max=info
                model_state=estimator.net.state_dict()
                optimizer_state=estimator.optimizer.state_dict()
            if (info!=info)==True:
                mi_mean_batch=mi_mean_batch/(train_step)
                print('final mi is {}'.format(mi_mean_batch))
                print('info_max is {}'.format(info_max))
                early_stop=1
                break 
            elif info_max!=0:
                estimator.net.load_state_dict(model_state)
                estimator.optimizer.load_state_dict(optimizer_state)
              
                
            writer.add_scalar('info',info,epoch)
            if args.resource=='test':
                writer.add_scalar('true info',true_mi_value,epoch)
            mi_mean_batch+=info
        if epoch==int(args.epoch)-1:
            mi_mean_batch=mi_mean_batch/(train_step)
            print('final mi is {}'.format(mi_mean_batch))
            print('info_max is {}'.format(info_max))
        if early_stop==1:
            break

                
    

if __name__ == '__main__':
    path='/home/declare/zodiark/Causal_Emergence_LLMs/data/simulation/gaussian_in__num_10000_dim_5.json'
    macro_miestimator(path,args.input_dimension,args.output_dimension)
    