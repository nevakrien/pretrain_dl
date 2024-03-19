import intel_extension_for_pytorch as ipex
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer,TrainingArguments,Trainer ,DataCollatorForLanguageModeling
from bigdl.llm.transformers import AutoModelForCausalLM#,Trainer,DataCollatorForLanguageModeling

model=AutoModelForCausalLM.from_pretrained("random_bert")#.to('xpu')
#model=torch.nn.DataParallel(model, device_ids=[0, 1, 2,3])

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenized_datasets = load_from_disk('dataset')
eval_dataset=load_from_disk('dataset_val')

print(tokenized_datasets)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,  # Enable MLM
    mlm_probability=0.15  # Set the masking probability to 15%
)

