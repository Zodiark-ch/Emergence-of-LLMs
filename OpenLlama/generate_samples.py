import torch
from torchsummary import summary
from random_sample_from_model import Sample_from_GPT2LMHEADMODEL,Sample_from_OpenAIGPTModel,Sample_from_GPTNeoXForCausalLM,Sample_from_LlamaForCausalLM
from random_sample_with_token_from_model import Sample_with_token
from random_sample_from_dataset import Sample_from_openllama_dataset,Sample_with_context
from transformers import HfArgumentParser
from args import DeepArgs
from utils import set_gpu 


hf_parser = HfArgumentParser((DeepArgs,))
args: DeepArgs = hf_parser.parse_args_into_dataclasses()[0]

set_gpu(args.gpu)
    
if args.task_name=='sample_from_context':
    model=Sample_from_openllama_dataset(args)
    
if args.task_name=='sample_from_dataset':
    model=Sample_with_context(args)

outputs = model()

