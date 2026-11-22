import cs336_basics.modules as m
import cs336_basics.bpe as bpe
from cs336_basics.config import *

lm = m.TransformerLM(d_model, num_heads, d_ff, vocab_size, context_len, num_layers, theta, device=device)
m.load_checkpoint("/Users/yuxlu/Documents/learning/assignment1-basics/data/tiny_stories/5000.ckpt", lm, None)
tokenizer = bpe.Tokenizer.from_files(bpe_vocab, bpe_merges, special_tokens)

input = "Meichen and I went to a park last night and "
print(input)
print(m.generate(input, tokenizer, lm, max_gen=256, temp=1, prob=0.6, context_len=context_len, device=device))


