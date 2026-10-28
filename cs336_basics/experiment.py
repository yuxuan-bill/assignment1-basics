import cs336_basics.bpe as bpe
from datetime import datetime

tokenizer = bpe.Tokenizer.from_files("/Users/yuxlu/Documents/learning/assignment1-basics/data/tiny_stories_vocab",
                         "/Users/yuxlu/Documents/learning/assignment1-basics/data/tiny_stories_merges", ['<|endoftext|>'])

with open("/Users/yuxlu/Documents/learning/assignment1-basics/data/owt_valid.txt") as f:
    t = f.read(10000000)
print(len(t))
print(datetime.now())
print(len(tokenizer.encode(t)))
print(datetime.now())