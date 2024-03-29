import torch
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch

from datasets import load_from_disk
from transformers import AutoTokenizer,TrainingArguments,Trainer ,DataCollatorForLanguageModeling
from bigdl.llm.transformers import AutoModelForCausalLM#,Trainer,DataCollatorForLanguageModeling

model=AutoModelForCausalLM.from_pretrained("random_hebert")#.to('xpu')
#model=torch.nn.DataParallel(model, device_ids=[0, 1, 2,3])

#model_name = "bert-base-uncased"
model_name="avichr/heBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenized_datasets = load_from_disk('HeNLP_HeDC4_dataset')
eval_dataset=load_from_disk('HeNLP_HeDC4_dataset_val')

print(tokenized_datasets)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,  # Enable MLM
    mlm_probability=0.15  # Set the masking probability to 15%
)


training_args = TrainingArguments(
    output_dir="./results",
    #xpu_backend='mpi',
    #bf16=True,
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    #fp16=True,  # If your GPUs support FP16.
    push_to_hub=False,
    logging_dir='./logs',
    use_ipex=True,
    #xpu_backend='mpi',
    evaluation_strategy="epoch",
)

print(len(tokenized_datasets))
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,#[100:],
    eval_dataset=eval_dataset,#tokenized_datasets[:100], # If you have a test split.
    #place_model_on_device=True,
)

trainer.train(resume_from_checkpoint=True)
trainer.save_model("pretrained_hebert")

