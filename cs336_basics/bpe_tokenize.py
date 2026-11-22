import cs336_basics.bpe as bpe
import numpy as np


def tokenize(tokenizer, file_to_tokenize):
    with open(file_to_tokenize) as f:
        with open(file_to_tokenize + ".tokens", 'wb') as token_f:
            token_ids_itr = tokenizer.encode_iterable(f)
            buffer = []
            for id in token_ids_itr:
                buffer.append(id)
                if len(buffer) > 1e6:
                    arr = np.array(buffer, dtype='<u2')
                    arr.tofile(token_f)
                    buffer = []
            arr = np.array(buffer, dtype='<u2')
            arr.tofile(token_f)

tokenizer = bpe.Tokenizer.from_files("/Users/yuxlu/Documents/learning/assignment1-basics/data/tiny_stories_vocab",
                         "/Users/yuxlu/Documents/learning/assignment1-basics/data/tiny_stories_merges", ['<|endoftext|>'])
tokenize(tokenizer, "/Users/yuxlu/Documents/learning/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt")
tokenize(tokenizer, "/Users/yuxlu/Documents/learning/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt")

tokenizer = bpe.Tokenizer.from_files("/Users/yuxlu/Documents/learning/assignment1-basics/data/owt_vocab",
                         "/Users/yuxlu/Documents/learning/assignment1-basics/data/owt_merges", ['<|endoftext|>'])
tokenize(tokenizer, "/Users/yuxlu/Documents/learning/assignment1-basics/data/owt_train.txt")
tokenize(tokenizer, "/Users/yuxlu/Documents/learning/assignment1-basics/data/owt_valid.txt")

'''
data = np.memmap(file_to_tokenize + ".tokens", dtype='<u2', mode='r')
print(data.shape)
print(data[:10])
'''
