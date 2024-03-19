import intel_extension_for_pytorch as ipex
from datasets import load_from_disk
from transformers import AutoTokenizer, Trainer,TrainingArguments ,DataCollatorForLanguageModeling
from bigdl.llm.transformers import AutoModelForCausalLM#,Trainer,DataCollatorForLanguageModeling

import torch
from time import time

t=time()
while(True):
    x=torch.ones(10000).to('xpu')
    x*=5

    if(time()-t > 30):
        break
    
