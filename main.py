from datasets import load_from_disk
from transformers import AutoTokenizer, Trainer,TrainingArguments ,DataCollatorForLanguageModeling
from bigdl.llm.transformers import AutoModelForCausalLM#,Trainer,DataCollatorForLanguageModeling

model=AutoModelForCausalLM.from_pretrained("random_bert")

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenized_datasets = load_from_disk('dataset')

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,  # Enable MLM
    mlm_probability=0.15  # Set the masking probability to 15%
)


training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=True,  # If your GPUs support FP16.
    push_to_hub=False,
    logging_dir='./logs',
)

print(len(tokenized_datasets))
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,#[100:],
    #eval_dataset=tokenized_datasets[:100], # If you have a test split.
)

trainer.train()
trainer.save_model("pretrained")
