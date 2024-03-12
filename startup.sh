#!/bin/bash

# sudo apt-get update

sudo apt-get install python3-pip -y

sudo pip3 install huggingface_hub[hf_transfer]
export HF_HUB_ENABLE_HF_TRANSFER=1

sudo pip3 install -r requirements.txt
sudo pip3 install transformers[torch]

sudo pip3 install git+https://github.com/georgeyw/lang.git
