# pretrain_dl

torchrun --nproc_per_node 4 hugging_train.py

# notes on structure
I have kept my failed attempts in the code base in case we would like to revisit them in the future.
the code that DOES work is

1. data.py (makes the dataset)
2. make_random_bert.py (makes and saves a randomly initilized huggingface model)
3. hugging_train.py (runs training using hugingface and bigdl)
4. train_hebert.py (a specific usage example thats curently runing)

i am inspired by the DPO example and the deepspeed example from the official repo (they both showed that trusting pytorch primitives works) 

the code should be able to take any raw pytorch model.


# recomanded ccl breaks
python -m pip install torch==2.0.1a0 torchvision==0.15.2a0 intel-extension-for-pytorch==2.0.110+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
