import sys
import tempfile


out_dir = 'out'
dataset = 'telegram'
eval_interval = 5
eval_iters = 30

init_from = 'gpt2-xl' # gpt2 variant (e.g. 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
device = 'cuda'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 30000

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False


# saving the parameters in a file in the Karpathy setup
config = {k: v for k, v in locals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))}
tmp = tempfile.NamedTemporaryFile(delete=False)
with open(tmp.name, 'w') as f:
    _ = [f.write('{} = "{}"\n'.format(k, v)) if isinstance(v, (str)) else f.write('{} = {}\n'.format(k, v)) for k, v in config.items()]


sys.argv = ['finetuning', tmp.name]
exec(open('train.py').read())
