#load Hugging Face Transformers model with INT4 optimizations
from bigdl.llm.transformers import AutoModelForCausalLM

model_path="microsoft/phi-2"
input_str="hello wolrd"
model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True)

#run the optimized model on Intel GPU
model = model.to('xpu')

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
input_ids = tokenizer.encode(input_str,return_tensors='pt').to('xpu')
output_ids = model.generate(input_ids)
output = tokenizer.batch_decode(output_ids.cpu())
print(output)
