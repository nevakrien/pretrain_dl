from datasets import load_dataset
from transformers import AutoTokenizer

if __name__ == "__main__":
    dataset = load_dataset('bookcorpus', split='train[1000:]')
    test_dataset=load_dataset('bookcorpus', split='train[:1000]')
    
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        # Replace 'text' with the correct field of your dataset
        return tokenizer(examples["text"],truncation=True, padding='max_length', max_length=128, return_special_tokens_mask=True)

    tokenized_datasets = test_dataset.map(tokenize_function, batched=True)
    tokenized_datasets.save_to_disk('dataset_val')

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.save_to_disk('dataset')

