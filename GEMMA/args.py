import os
import pickle
from typing import List
from dataclasses import field, dataclass
from utils import set_default_to_empty_string

FOLDER_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))+'/autodl-tmp/data/sample_from_gpt2-large/'


@dataclass
class DeepArgs:
    task_name: str = "sample_from_context"#'sample_with_token','sample_from_random','sample_from_dataset'
    model_name: str = "gpt2large"#"gptj""gpt2lmheadmodel","gpt1","gptneox""gpt2large""gemma-2b"
    token_num: int = 8
    sample_size: int = 60
    device: str = 'cuda:0'
    batch_size: int = 5000
    save_folder: str = os.path.join(FOLDER_ROOT, task_name,'token_num'+str(token_num))
    using_old: bool = False
    ablation_attn=False
    ablation_LN=False


    def __post_init__(self):
        
        assert self.task_name in ['sample_from_random', 'sample_from_dataset', 'sample_from_context', 'emo']
        assert self.model_name in ['gptneox','gpt2lmheadmodel', 'gpt-j-6b','gpt2large','gemma-2b']
        assert 'cuda:' in self.device
        self.gpu = int(self.device.split(':')[-1])
        self.actual_sample_size = self.sample_size


    def load_result(self):
        with open(self.save_file_name, 'rb') as f:
            return pickle.load(f)

