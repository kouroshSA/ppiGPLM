# Training config: GPT-2 small (124M) from scratch on character-level protein-pair prompts (MED4 PPI).

out_dir = 'out_14e'
eval_interval = 250 # keep frequent because we'll overfit
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = True
# override via command line if you like
wandb_project = 'ppiGLM_MED4Solo'
wandb_run_name = 'ppiGPLM-med4_4k_14epoch'
dataset = 'med4_4k_14epoch'
init_from = 'scratch'  # train from random init; no pretrained GPT-2 weights loaded
gradient_accumulation_steps = 2
batch_size = 12

block_size = 4096 # context of up to n previous characters


# GPT2-M models
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.2

# using above parameters, gradient = 10, batch = 16, token/iter = 98304, epoch = 97600


learning_rate = 5e-4 # with baby networks can afford to go a bit higher
max_iters = 8001
lr_decay_iters = 8000 # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 200 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

# To tokenize the training data:  python data/MED4_char/prepare.py
#
# To train GPT-2 small from scratch on a single GPU:
#   python train_.py config/train_par_gpt2-s_scratch.py
#
# To train on 2 GPUs with DDP:
#   torchrun --standalone --nproc_per_node=2 train_.py config/train_par_gpt2-s_scratch.py
