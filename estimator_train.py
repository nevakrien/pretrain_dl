from datasets import load_from_disk
from transformers import TFAutoModelForMaskedLM,TrainingArguments,Trainer ,DataCollatorForLanguageModeling
#from bigdl.llm.transformers import AutoModelForCausalLM#,Trainer,DataCollatorForLanguageModeling

from bigdl.dllib.keras.objectives import SparseCategoricalCrossEntropy
from bigdl.dllib.optim.optimizer import Adam
from bigdl.orca.learn.bigdl import Estimator

print('user code')

model=TFAutoModelForMaskedLM.from_pretrained("random_bert_tf")#.to('xpu')
#model=torch.nn.DataParallel(model, device_ids=[0, 1, 2,3])

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenized_datasets = load_from_disk('dataset').select(range(1000))
eval_dataset=load_from_disk('dataset_val')

print(tokenized_datasets)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,  # Enable MLM
    mlm_probability=0.15  # Set the masking probability to 15%
)

est = Estimator.from_bigdl(model=model, loss=SparseCategoricalCrossEntropy(), optimizer=Adam())

print(len(tokenized_datasets))
#trainer = Trainer(
#    model=model,
#    args=training_args,
#    data_collator=data_collator,
#    train_dataset=tokenized_datasets,#[100:],
#    eval_dataset=eval_dataset,#tokenized_datasets[:100], # If you have a test split.
#)

est.fit(tokenized_datasets, 1, batch_size=256)
print(est.evaluate(tokenized_datasets, 1, batch_size=256))
 
model.save_pretrained('model_estimator_bigdl')
