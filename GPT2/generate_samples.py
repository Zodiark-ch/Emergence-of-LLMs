import torch,os
from torchsummary import summary
from random_sample_from_model import Sample_from_GPT2LMHEADMODEL,Sample_from_OpenAIGPTModel,Sample_from_GPTNeoXForCausalLM
from random_sample_from_dataset import Sample_with_context_GPT2XL,Sample_with_context_GPT2large,Sample_with_context_Gemma,Sample_with_context_Openllama, \
    Sample_with_sequence_GPT2XL,Sample_with_sequence_GPT2large,Sample_with_sequence_Gemma,Sample_with_sequence_Openllama
from transformers import HfArgumentParser
from args import DeepArgs
from utils import set_gpu 


hf_parser = HfArgumentParser((DeepArgs,))
args: DeepArgs = hf_parser.parse_args_into_dataclasses()[0]

set_gpu(args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
if args.task_name=='sample_from_random':
    if args.model_name=='gpt2lmheadmodel':
        model=Sample_from_GPT2LMHEADMODEL(args)

    if args.model_name=='gpt1':
        model=Sample_from_OpenAIGPTModel(args)
        
    if args.model_name=='gptneox':
        model=Sample_from_GPTNeoXForCausalLM(args)
        
    if args.model_name=='openllama3b':
        model=Sample_from_LlamaForCausalLM(args)
    
if args.task_name=='sample_from_sequence':
    if args.model_name=='gpt2lmheadmodel':
        model=Sample_with_sequence_GPT2XL(args)
    if args.model_name=='gpt2large':
        model=Sample_with_sequence_GPT2large(args)
    if args.model_name=='gemma':
        model=Sample_with_sequence_Gemma(args)
    if args.model_name=='openllama':
        model=Sample_with_sequence_Openllama(args)
    
if args.task_name=='sample_from_context':
    if args.model_name=='gpt2lmheadmodel':
        model=Sample_with_context_GPT2XL(args)
    if args.model_name=='gpt2large':
        model=Sample_with_context_GPT2large(args)
    if args.model_name=='gemma':
        model=Sample_with_context_Gemma(args)
    if args.model_name=='openllama':
        model=Sample_with_context_Openllama(args)
    
outputs = model()

