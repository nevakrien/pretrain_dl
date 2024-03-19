import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from bigdl.llm.transformers import AutoModelForCausalLM
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling


# Assuming the necessary imports from PyTorch and other libraries are already done

# Initialize model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained("random_bert").to('xpu')

model= torch.nn.DataParallel(model, device_ids=[0, 1, 2,3])

# Load datasets
train_dataset = load_from_disk('dataset')
eval_dataset = load_from_disk('dataset_val')

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])



# Initialize the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

print(train_dataset[0])
train_dataset=train_dataset[:1000]

# Prepare DataLoader
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32, data_collator)
eval_dataloader = DataLoader(eval_dataset, batch_size=32, collate_fn=data_collator)

print(next(iter(train_dataloader)))

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
num_train_epochs = 3
for epoch in range(num_train_epochs):
    model.train()
    for batch in train_dataloader:
        #batch = {k: v.to('xpu') for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Evaluation loop
    model.eval()
    total_eval_loss = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            #batch = {k: v.to('xpu') for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_eval_loss += loss.item()
    
    print(f"Epoch {epoch+1} completed. Avg eval loss: {total_eval_loss / len(eval_dataloader)}")

# Save the model
model.save_pretrained("pretrained")

