import os,torch
from torchsummary import summary
from random_sample_from_model import Sample_from_GPT2LMHEADMODEL,Sample_from_OpenAIGPTModel,Sample_from_GPTNeoXForCausalLM,Sample_from_GEMMA2B
from random_sample_from_dataset import Sample_from_GEMMA2B_dataset,Sample_with_context,Sample_with_context_GPT2large
from transformers import HfArgumentParser
from args import DeepArgs
from utils import set_gpu 


hf_parser = HfArgumentParser((DeepArgs,))
args: DeepArgs = hf_parser.parse_args_into_dataclasses()[0]

set_gpu(args.gpu)

if args.task_name=='sample_from_dataset':
    model=Sample_with_context(args)
        
if args.task_name=='sample_from_context':
    model=Sample_from_GEMMA2B_dataset(args)

outputs = model()


