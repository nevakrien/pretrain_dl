# pretrain_dl

torchrun --nproc_per_node 4 hugging_train.py

# recomanded ccl breaks
python -m pip install torch==2.0.1a0 torchvision==0.15.2a0 intel-extension-for-pytorch==2.0.110+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
