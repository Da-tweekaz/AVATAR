# AVATAR
This repo is for source code of paper "Aligning Visual and Textual Semantics for Generative Recommendation".

# Data Preprocess
Step 1: Decompress the downloaded 5-core reviews and metadata from Amazon Review 2014, which are in the format `reviews_Beauty_5.json.gz` and `meta_Beauty.json.gz`. Use the command provided in the `AVATAR/data/process.ipynb` file to perform the decompression.

Step 2: Use `main.py` in `AVATAR/rqvae` folder to train a rqvae model using semantic embeddings obtained in Step 1.

Step 3: Use `generate_code.py` in `AVATAR/rqvae` folder to select the best model to generate discrete code for semantic embeddings in Step 1 and padding at the last position to resolve duplicate codes.

# Pre-training
Use `pretrain.py` in `AVATAR/model` folder to pre-train a T5 encoder-decoder model with semantic code.

# Fine-tuning
Use `finetune.py` in `AVATAR/model` folder to fine-tune a T5 encoder-decoder model with semantic code.

# Acknowledgement
The structure of this code is based on [TIGER](https://github.com/XiaoLongtaoo/TIGER). Thank for their work.
