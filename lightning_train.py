import os
from typing import Optional

from bigdl.nano.pytorch import Trainer
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

class MLMDataModule(LightningDataModule):
    def __init__(self, data_dir: str, tokenizer: AutoTokenizer, data_collator: DataCollatorForLanguageModeling):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.data_collator = data_collator

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = load_from_disk(os.path.join(self.data_dir, 'dataset'))
        self.val_dataset = load_from_disk(os.path.join(self.data_dir, 'dataset_val'))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, collate_fn=self.data_collator)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32, collate_fn=self.data_collator)

class MLMModel(LightningModule):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15,
        )
        self.val_batch_count = 0

    def forward(self, input_ids, labels):
        return self.model(input_ids=input_ids, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)

        # Run validation after every 10,000 training batches
        if (batch_idx + 1) % 10000 == 0:
            self.run_validation(batch_idx)

        return loss

    def run_validation(self, batch_idx):
        self.val_batch_count = 0
        val_losses = []
        val_loader = self.trainer.datamodule.val_dataloader()
        for val_batch in val_loader:
            val_outputs = self(**val_batch)
            val_loss = val_outputs.loss
            val_losses.append(val_loss)
            self.val_batch_count += 1

        val_loss = torch.stack(val_losses).mean()
        self.log("val_loss", val_loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-5)

# Set up the data module
data_module = MLMDataModule(data_dir=".", tokenizer=tokenizer, data_collator=data_collator)

# Set up the model
model = MLMModel("bert-base-uncased")

# Set up the trainer
trainer = Trainer(
    default_root_dir="./results",
    max_epochs=3,
    gradient_clip_val=1.0,
    accelerator="xpu",
    precision=16 if torch.cuda.is_available() else 32,
)

# Train the model
trainer.fit(model, data_module)

# Save the model
trainer.save_model("pretrained")
