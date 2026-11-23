d_model = 512
num_heads = 16
d_ff = 1344
vocab_size = 10000
context_len = 256
num_layers = 4
theta = 10000
batch_size = 32

total_steps = 5001
weight_decay = 0.1
eps = 1e-6
betas = (0.9, 0.999)
max_lr = 0.006
min_lr = 0.002
warmup_itrs = 1000
cosine_itrs = total_steps - warmup_itrs
max_l2_norm = 1.0


device = "mps"


total_tokens = total_steps * context_len * batch_size
ckpt_intervals = 1000

load_from = None
#load_from = "/Users/yuxlu/Documents/learning/assignment1-basics/data/tiny_stories/50.ckpt"

config = {
    "d_model": d_model,
    "num_heads": num_heads,
    "d_ff": d_ff,
    "vocab_size": vocab_size,
    "context_len": context_len,
    "num_layers": num_layers,
    "theta": theta,
    "batch_size": batch_size,
    "total_tokens": total_tokens,
    "ckpt_intervals": ckpt_intervals,

    "weight_decay": weight_decay,
    "eps": eps,
    "betas": betas,
    "max_lr": max_lr,
    "min_lr": min_lr,
    "warmup_itrs": warmup_itrs,
    "cosine_itrs": cosine_itrs,
    "max_l2_norm": max_l2_norm,
}

ckpt_folder = "/Users/yuxlu/Documents/learning/assignment1-basics/data/tiny_stories"
train_token_file = "/Users/yuxlu/Documents/learning/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt.tokens"
validation_token_file = "/Users/yuxlu/Documents/learning/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt.tokens"

bpe_merges = "/Users/yuxlu/Documents/learning/assignment1-basics/data/tiny_stories_merges"
bpe_vocab = "/Users/yuxlu/Documents/learning/assignment1-basics/data/tiny_stories_vocab"

'''
ckpt_folder = "/workspace/data/tiny_stories"
train_token_file = "/workspace/data/TinyStoriesV2-GPT4-train.txt.tokens"
validation_token_file = "/workspace/data/TinyStoriesV2-GPT4-valid.txt.tokens"

bpe_merges = "/workspace/data/tiny_stories_merges"
bpe_vocab = "/workspace/data/tiny_stories_vocab"
'''
special_tokens = ["<|endoftext|>"]


