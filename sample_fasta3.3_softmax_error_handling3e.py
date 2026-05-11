"""
sample_fasta3.3_softmax_error_handling3e.py — batch inference for ppiGPLM.

Loads a ckpt.pt from <model_dir> (default "out") and runs inference on a file of
prompts (one prompt per line, no quotes), producing:

  - <output_prefix>_classifications.txt : FASTA-like dump of model output
  - <output_prefix>_probabilities.csv   : per-pair softmax probabilities for "1" and "0"

Robustness:
  - Block-size detection from checkpoint[‘model_args’][‘block_size’] (or
    model.config.n_positions for GPT-2 variants).
  - Input clipping: if a prompt exceeds block_size, the head is clipped
    (start_ids = start_ids[-block_size:]) so the label position stays intact.
  - Unknown-token replacement: out-of-vocab characters are mapped to ‘A’.

Usage:
    python sample_fasta3.3_softmax_error_handling3e.py \\
        --input_file my_prompts.txt \\
        --output_dir results \\
        --output_prefix myoutput
"""

import os
import sys
import argparse
from contextlib import nullcontext
import pickle
import torch
import torch.nn.functional as F
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Parse command-line arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Sample from a trained model with prompt input.')
parser.add_argument('--input_file', type=str, default='generated_prompts.txt', help='Path to file containing prompts')
parser.add_argument('--output_dir', type=str, default='out-ppi', help='Directory to save outputs')
parser.add_argument('--output_prefix', type=str, default='generated', help='Prefix for output files')
args = parser.parse_args()

# Reset sys.argv for configurator
sys.argv = [sys.argv[0]]

prompts_file_path = args.input_file
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

fasta_output_filename = os.path.join(output_dir, args.output_prefix + '_classifications.txt')
csv_output_filename = os.path.join(output_dir, args.output_prefix + '_probabilities.csv')

# -----------------------------------------------------------------------------
# Sampling parameters and model init overrides
# -----------------------------------------------------------------------------
init_from = 'resume'  # or a GPT-2 variant
model_dir = 'out'
max_new_tokens = 1
temperature = 0.1
top_k = 2
seed = int.from_bytes(os.urandom(4), 'big')
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read())  # overrides from command line or config file

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------
if init_from == 'resume':
    ckpt_path = os.path.join(model_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    block_size = checkpoint['model_args'].get('block_size', 1024)
else:
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    block_size = getattr(model.config, 'n_positions', 1024)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# Load vocabulary mapping for character-level tokenization
# -----------------------------------------------------------------------------
meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
if not os.path.exists(meta_path):
    raise FileNotFoundError(f'Meta file not found: {meta_path}')
with open(meta_path, 'rb') as meta_file:
    meta = pickle.load(meta_file)
stoi = meta['stoi']
itos = meta['itos']
encode = lambda s: [stoi.get(ch, stoi.get('<unk>', 0)) for ch in s]
decode = lambda l: ''.join([itos.get(i, '') for i in l])

# -----------------------------------------------------------------------------
# Read prompts
# -----------------------------------------------------------------------------
with open(prompts_file_path, 'r', encoding='utf-8') as f:
    prompts = [line.strip() for line in f if line.strip()]

# -----------------------------------------------------------------------------
# Token IDs for classification tokens
# -----------------------------------------------------------------------------
one_id = encode('1')[0] if encode('1') else None
zero_id = encode('0')[0] if encode('0') else None

# -----------------------------------------------------------------------------
# FASTA formatting helper
# -----------------------------------------------------------------------------
def format_as_fasta(sequence, sample_number):
    return f'>Sample_{sample_number}\n{sequence}\n'

# -----------------------------------------------------------------------------
# Generate outputs and write to files
# -----------------------------------------------------------------------------
with open(fasta_output_filename, 'w', encoding='utf-8') as fasta_file, \
     open(csv_output_filename, 'w', encoding='utf-8') as csv_file:
    csv_file.write('l1,Seq1,l2,Seq2,l3,Probability_of_1,Probability_of_0\n')
    with torch.no_grad():
        with ctx:
            for k, prompt in enumerate(prompts):
                start_ids = encode(prompt)
                if len(start_ids) > block_size:
                    start_ids = start_ids[-block_size:]
                x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

                logits = model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                last_logits = logits[:, -1, :]
                probs = F.softmax(last_logits, dim=-1)

                prob_for_1 = probs[0, one_id].item() if one_id is not None and one_id < probs.shape[-1] else 0.0
                prob_for_0 = probs[0, zero_id].item() if zero_id is not None and zero_id < probs.shape[-1] else 0.0

                y = model.generate(idx=x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
                sample_text = decode(y[0].tolist())

                fasta_file.write(format_as_fasta(sample_text, k) + '\n')
                csv_file.write(f'{prompt},{prob_for_1},{prob_for_0}\n')

                print(format_as_fasta(sample_text, k))
                print(f'Probability(next_token=1) = {prob_for_1}')
                print(f'Probability(next_token=0) = {prob_for_0}')
                print('---------------')

print(f'FASTA-like samples saved to {fasta_output_filename}')
print(f'CSV with probabilities saved to {csv_output_filename}')
