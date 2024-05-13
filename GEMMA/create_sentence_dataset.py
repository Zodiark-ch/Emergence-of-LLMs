import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM,GPT2LMHeadModel,LlamaTokenizer,LlamaForCausalLM
import argparse


# inputs = tokenizer(')', return_tensors="pt")
# print(inputs["input_ids"])
#. is with token_ids=13   : is 25, /n is 198 ? is 30, ; is 26 - is 12. () is 7 and 8

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gemma', type= str, help='gpt2,openllama,gemma')
parser.add_argument('--type', type=str, default='0',  help='0 represent token0 is close to id13, 1represent token-1 is close to id13')
parser.add_argument('--dataset',type=str, default='hermes',help='orca,hermes')
args = parser.parse_args()

if args.model=='gpt2':
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model =GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    
if args.model=='openllama':
    tokenizer =LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b")
    model =LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b")
    
if args.model=='gemma':
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
    
inputs = tokenizer('I like it.', return_tensors="pt")
print(inputs["input_ids"])
inputs = tokenizer('I like it?', return_tensors="pt")
print(inputs["input_ids"])
inputs = tokenizer('I like it:', return_tensors="pt")
print(inputs["input_ids"])
inputs = tokenizer('I like it/n', return_tensors="pt")
print(inputs["input_ids"])
inputs = tokenizer('I like it-', return_tensors="pt")
print(inputs["input_ids"])

#. is with token_ids=235265   : is 235292, /n is 235283 ? is 235336, - is 235290.
    
    
if args.dataset=='orca':
    ds = load_dataset("Open-Orca/OpenOrca")
    data=ds['train']
    data_length=len(data)
    sample_sequeece=np.random.choice(data_length,1000000)
    
if args.dataset=='hermes':
    ds = load_dataset("teknium/OpenHermes-2.5")
    data=ds['train']['conversations']
    data_length=len(data)
    sample_sequeece=np.random.choice(data_length,1000000)

num=0
batch_squeeze=torch.ones((1,16))
for i in range(len(sample_sequeece)):
    data_id=int(sample_sequeece[i])
    if args.dataset=='orca':
        question=data[data_id]['question']
    if args.dataset=='hermes':
        question=data[data_id][1]['value']
    inputs = tokenizer(question, return_tensors="pt")
    input_ids=inputs["input_ids"][0]
    if input_ids.size()[-1]>20:
        
        length=input_ids.size()[0]
        idx=torch.nonzero(input_ids==235265)
        if idx.size()[0]>1:
            sample_flag=1
            for i_idx in range(idx.size()[0]):
                if sample_flag==0:
                    break
                if i_idx+2>idx.size()[0]:
                    squeeze_length=length-idx[i_idx][0].item()
                else:
                    squeeze_length=idx[i_idx+1][0].item()-idx[i_idx][0].item()
                if squeeze_length>=17:
                    flag=1
                    if args.type=='0':
                        squeeze=input_ids[idx[i_idx][0]+1:idx[i_idx][0]+17]
                    elif args.type=='1' and i_idx+2<=idx.size()[0]:
                        squeeze=input_ids[idx[i_idx+1][0]-16:idx[i_idx+1][0]]
                    elif args.type=='1' and i_idx+2>idx.size()[0]:
                        squeeze=input_ids[length-16:]
                    else:
                        break
                    assert squeeze.size()[0]==16
                    for i_check in range(16):
                        if squeeze[i_check].item()==235265 or squeeze[i_check].item()==235292 or squeeze[i_check].item()==235283 or squeeze[i_check].item()==235336 or squeeze[i_check].item()==235290:
                            flag=0 
                            break 
                    if flag==1:
                        sample_flag=0
                        #squeeze=squeeze.unsqueeze(0)
                        # print(squeeze)
                        # print(tokenizer.convert_ids_to_tokens(squeeze))
                        squeeze=squeeze.unsqueeze(0)
                        if batch_squeeze[0][1].item()==1 and batch_squeeze.size()[0]==1:
                            batch_squeeze=squeeze
                        elif batch_squeeze.size()[0]<500: 
                            batch_squeeze=torch.cat((batch_squeeze,squeeze),dim=0)
                            # print(i,data_id)
                        elif batch_squeeze.size()[0]==500 and num<600:
                            print('save {} batch5000 tensor'.format(num),batch_squeeze.size()) 
                            torch.save(batch_squeeze,'data/MIdataset/context_gpt_tensor/{}_batch500_OpenHermes.pth'.format(num))
                            
                            num=num+1
                            batch_squeeze=torch.ones((1,16))
                        else:
                            break
                        
        
                