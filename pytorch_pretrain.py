
import torch
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch

import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from bigdl.llm.transformers import AutoModelForMaskedLM
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling
import os

from tqdm import tqdm
# Assuming the necessary imports from PyTorch and other libraries are already done

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['CCL_ZE_IPC_EXCHANGE'] = 'sockets'
os.environ['CCL_ATL_TRANSPORT']='mpi'
# initialize the process group
rank=0
dist.init_process_group("ccl", rank=rank, world_size=1)
device=torch.device(f'xpu:{rank}')

# Initialize model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained("random_bert").to(device)#('xpu')

#model= torch.nn.DataParallel(model, device_ids=[0, 1, 2,3])

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
train_dataset=train_dataset.select(range(1000))#[:1000]

# Prepare DataLoader
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32, collate_fn=data_collator)
eval_dataloader = DataLoader(eval_dataset, batch_size=32, collate_fn=data_collator)

print(next(iter(train_dataloader)))

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
num_train_epochs = 3
for epoch in range(num_train_epochs):
    model.train()
    train_loss = 0.0
    tqdm_train_loader = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
    for batch in tqdm_train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        train_loss += loss.item()
        # Format loss to display with 4 decimal places
        current_train_loss = train_loss / (tqdm_train_loader.n + 1)
        tqdm_train_loader.set_postfix({"train_loss": f"{current_train_loss:.4f}"})
    
    # Evaluation loop
    model.eval()
    total_eval_loss = 0
    tqdm_eval_loader = tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch+1}")
    with torch.no_grad():
        for batch in tqdm_eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            total_eval_loss += loss.item()
            # Format eval loss to display with 4 decimal places
            current_eval_loss = total_eval_loss / (tqdm_eval_loader.n + 1)
            tqdm_eval_loader.set_postfix({"eval_loss": f"{current_eval_loss:.4f}"})

    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    print(f"Epoch {epoch+1} completed. Avg eval loss: {avg_eval_loss:.4f}")


# Save the model
model.save_pretrained("pretrained")

