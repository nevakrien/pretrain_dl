from datasets import load_dataset
from transformers import AutoTokenizer, Trainer,TrainingArguments ,DataCollatorForLanguageModeling
from bigdl.llm.transformers import AutoModelForCausalLM#,Trainer,DataCollatorForLanguageModeling

model=AutoModelForCausalLM.from_pretrained("random_bert")

dataset = load_dataset('bookcorpus')

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    # Replace 'text' with the correct field of your dataset
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")

tokenized_datasets = dataset.map(tokenize_function, batched=True)

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

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"], # If you have a test split.
)

trainer.train()
trainer.save_model("finetuned")
