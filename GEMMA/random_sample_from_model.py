import torch
import torch.nn as nn
import torch.nn.functional as F 
from transformers import AutoTokenizer, GPT2LMHeadModel,OpenAIGPTModel,OpenAIGPTTokenizer,GPTNeoXForCausalLM,AutoModelForCausalLM
from torchsummary import summary
from tqdm import trange
import numpy as np

class Sample_from_GPT2LMHEADMODEL(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.model_name=args.model_name
        self.sample_size=args.sample_size
        self.task_name=args.task_name
        self.token_num=args.token_num
        self.batch_size=args.batch_size
        
        self.model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.layers=self.model.transformer.h
        if args.ablation_attn==True:
            self.layers.apply(self.weights_init_attn)
        if args.ablation_LN==True:
            self.layers.apply(self.weights_init_linear)
        self.device=args.device
            
    @property
    def device(self):
        return self.model.device

    @device.setter
    def device(self, device):
        print(f'Model: set device to {device}')
        self.model = self.model.to(device)
        self.layers = self.layers.to(device)

            
    def weights_init_attn(self,m):

        classname = m.__class__.__name__
        if classname.find('GPT2Block') != -1:
            nn.init.xavier_normal_(m.attn.c_attn.weight.data)   
            nn.init.constant_(m.attn.c_attn.bias.data, 0.0)

    def weights_init_linear(self,m):

        classname = m.__class__.__name__
        if classname.find('GPT2Block') != -1:
            nn.init.constant_(m.ln_1.weight.data, 0.0)   
            nn.init.constant_(m.ln_1.bias.data, 0.0)
            nn.init.constant_(m.ln_2.weight.data, 0.0)   
            nn.init.constant_(m.ln_2.bias.data, 0.0)

        
    
            
    def forward(self,past_key_values = None,encoder_hidden_states= None,encoder_attention_mask = None,):
        inputs = self.tokenizer("Hello, my dog is cute", return_tensors="pt").to(self.device)
        outputs =self.model(**inputs, labels=inputs["input_ids"])
        text=self.get_text(self.token_num)
        inputs = self.tokenizer(text, return_tensors="pt")
        assert inputs["attention_mask"].size()[-1]==self.token_num
        attention_mask=inputs["attention_mask"]
        attention_mask=attention_mask.expand(self.batch_size,-1)
        batch_size=attention_mask.size()[0]
        assert batch_size==self.batch_size
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=torch.float32)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        head_mask = head_mask = [None] * 12
        for s in trange(self.sample_size):
            if past_key_values is None:
                past_length = 0
                past_key_values = tuple([None] * len(self.layers))
                
                
            inputs = self.tokenizer("Hello, my dog is cute", return_tensors="pt")
            for i, (block, layer_past) in enumerate(zip(self.layers, past_key_values)):
                
                x_sample = self.gen_x(batch_size,self.token_num, 768, 0.81)
                hidden_states = torch.tensor(x_sample,dtype=torch.float32).to(self.device)
        
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                #print(hidden_states.cpu())
                if i==7 or i==8:
                    input_save_dir='/sample_'+str(s)+'input_layer'+str(i)+"_token"+str(self.token_num)+'_batch'+str(self.batch_size)+'.pth'
                    torch.save(hidden_states.cpu(),self.args.save_folder+input_save_dir)
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=True,
                    output_attentions=False,
                )
                hidden_states = outputs[0]
                if i==7 or i==8:
                    output_save_dir='/sample_'+str(s)+'output_layer'+str(i)+"_token"+str(self.token_num)+'_batch'+str(self.batch_size)+'.pth'
                    torch.save(hidden_states.cpu(),self.args.save_folder+output_save_dir)
        return 1
    
    def get_text(self,num):
        text=''
        for i in range(num-1):
            text=text+str(i)+' '
        return text
    
    def gen_x(self,num, token_num,dim ,power):
        return np.random.normal(0., np.sqrt(power), [num,token_num, dim])
        



class Sample_from_OpenAIGPTModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.model_name=args.model_name
        self.sample_size=args.sample_size
        self.task_name=args.task_name
        self.token_num=args.token_num
        self.batch_size=args.batch_size
        
        self.model = OpenAIGPTModel.from_pretrained("openai-gpt")
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
        self.layers=self.model.h
        self.device=args.device

    @property
    def device(self):
        return self.model.device

    @device.setter
    def device(self, device):
        print(f'Model: set device to {device}')
        self.model = self.model.to(device)
        self.layers = self.layers.to(device)


        
    
            
    def forward(self,past_key_values = None,output_hidden_states= None,output_attentions = None,):
        text=self.get_text(self.token_num)
        inputs = self.tokenizer(text, return_tensors="pt")
        assert inputs["attention_mask"].size()[-1]==self.token_num
        attention_mask=inputs["attention_mask"]
        attention_mask=attention_mask.expand(self.batch_size,-1)
        batch_size=attention_mask.size()[0]
        assert batch_size==self.batch_size
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=torch.float32)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        head_mask = head_mask = [None] * 12
        for s in trange(self.sample_size):
            if past_key_values is None:
                past_length = 0
                past_key_values = tuple([None] * len(self.layers))
                
                
            inputs = self.tokenizer("Hello, my dog is cute", return_tensors="pt")
            for i, block in enumerate(self.layers):
                
                x_sample = self.gen_x(batch_size,self.token_num, 768, 0.81)
                hidden_states = torch.tensor(x_sample,dtype=torch.float32).cuda()
        
                #print(hidden_states.cpu())

                input_save_dir='/sample_'+str(s)+'input_layer'+str(i)+"_token"+str(self.token_num)+'_batch'+str(self.batch_size)+'.pth'
                torch.save(hidden_states.cpu(),self.args.save_folder+input_save_dir)
                attention_mask = attention_mask.to(hidden_states.device)
                outputs = block(hidden_states, attention_mask, head_mask[i], output_attentions=output_attentions)
                hidden_states = outputs[0]

                output_save_dir='/sample_'+str(s)+'output_layer'+str(i)+"_token"+str(self.token_num)+'_batch'+str(self.batch_size)+'.pth'
                torch.save(hidden_states.cpu(),self.args.save_folder+output_save_dir)
        return 1
    
    def get_text(self,num):
        text=''
        for i in range(num):
            text=text+str(i)+' '
        return text
    
    def gen_x(self,num, token_num,dim ,power):
        return np.random.normal(0., np.sqrt(power), [num,token_num, dim])
    
    


class Sample_from_GPTNeoXForCausalLM(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.model_name=args.model_name
        self.sample_size=args.sample_size
        self.task_name=args.task_name
        self.token_num=args.token_num
        self.batch_size=args.batch_size
        #GPTNeoXConfig.hidden_size=768
        
        self.model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b").half()
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

        self.layers=self.model.gpt_neox.layers
        self.device=args.device
            
    @property
    def device(self):
        return self.model.device

    @device.setter
    def device(self, device):
        print(f'Model: set device to {device}')
        self.model = self.model.to(device)
        self.layers = self.layers.to(device)


        
    
            
    def forward(self,past_key_values = None,inputs_embeds=None,position_ids= None,encoder_attention_mask = None,):
        # inputs = self.tokenizer("Hello, my dog is cute", return_tensors="pt").to(self.device)
        # outputs =self.model(**inputs, labels=inputs["input_ids"])
        text=self.get_text(self.token_num)
        inputs = self.tokenizer(text, return_tensors="pt")
        assert inputs["attention_mask"].size()[-1]==self.token_num
        attention_mask=inputs["attention_mask"]
        attention_mask=attention_mask.expand(self.batch_size,-1)
        batch_size=attention_mask.size()[0]
        assert batch_size==self.batch_size
        batch_size=batch_size
        seq_length = attention_mask.size()[1]
        past_length = 0
    
        if position_ids is None:
            position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0)
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=torch.float16)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float16).min
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        head_mask = head_mask = [None] * len(self.layers)
        for s in trange(self.sample_size):
            if past_key_values is None:
                past_length = 0
                past_key_values = tuple([None] * len(self.layers))
                
                
            inputs = self.tokenizer("Hello, my dog is cute", return_tensors="pt")
            for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
                
                x_sample = self.gen_x(batch_size,self.token_num, 6144, 0.81)
                hidden_states = torch.tensor(x_sample,dtype=torch.float16).to(self.device)
        
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                #print(hidden_states.cpu())
                if i==5:
                    input_save_dir='/layer_'+str(i)+'/sample_'+str(s)+'input_layer'+str(i)+"_token"+str(self.token_num)+'_batch'+str(self.batch_size)+'.pth'
                    torch.save(hidden_states.cpu(),self.args.save_folder+input_save_dir)
                outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    layer_past=layer_past,
                    use_cache=False,
                    output_attentions=True,
                )
                hidden_states = outputs[0]
                if i==5:
                    output_save_dir='/layer_'+str(i)+'/sample_'+str(s)+'output_layer'+str(i)+"_token"+str(self.token_num)+'_batch'+str(self.batch_size)+'.pth'
                    torch.save(hidden_states.cpu(),self.args.save_folder+output_save_dir)
        return 1
    
    def get_text(self,num):
        text=''
        for i in range(num-1):
            text=text+str(i)+' '
        return text
    
    def gen_x(self,num, token_num,dim ,power):
        return np.random.normal(0., np.sqrt(power), [num,token_num, dim])
        
        
class Sample_from_GEMMA2B(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.model_name=args.model_name
        self.sample_size=args.sample_size
        self.task_name=args.task_name
        self.token_num=args.token_num
        self.batch_size=args.batch_size
        
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
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
        inputs = self.tokenizer("Hello, my dog is cute", return_tensors="pt").to(self.device)
        outputs =self.model(**inputs, labels=inputs["input_ids"])
        text=self.get_text(self.token_num)
        inputs = self.tokenizer(text, return_tensors="pt")
        assert inputs["attention_mask"].size()[-1]==self.token_num
        attention_mask=inputs["attention_mask"].to(self.device)
        attention_mask=attention_mask.expand(self.batch_size,-1)
        batch_size=attention_mask.size()[0]
        assert batch_size==self.batch_size
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        head_mask = head_mask = [None] * 12
        for s in trange(self.sample_size):
                
            i=0    
            
            for decoder_layer in self.layers:
            
                x_sample = self.gen_x(batch_size,1, 2048, 0.81)
                inputs_embeds = torch.tensor(x_sample,dtype=torch.float32).to(self.device)
                inputs_embeds=inputs_embeds.expand(-1,self.token_num,-1)

                #print(hidden_states.cpu())
                if i==3 or i==4 or i==5: 
                    input_save_dir='/sample_'+str(s)+'input_layer'+str(i)+"_token"+str(self.token_num)+'_batch'+str(self.batch_size)+'.pth'
                    torch.save(inputs_embeds.cpu(),self.args.save_folder+input_save_dir)
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
                hidden_states = layer_outputs[0]
                if i==3 or i==4 or i==5:
                    output_save_dir='/sample_'+str(s)+'output_layer'+str(i)+"_token"+str(self.token_num)+'_batch'+str(self.batch_size)+'.pth'
                    torch.save(hidden_states.cpu(),self.args.save_folder+output_save_dir)
                i=i+1
        return 1
    
    def get_text(self,num):
        text=' '
        for i in range(int(num-(0.5*num)-1)):
            text=text+str(i)+' '
        return text
    
    def gen_x(self,num, token_num,dim ,power):
        return np.random.normal(0., np.sqrt(power), [num,token_num, dim])
    

        


        