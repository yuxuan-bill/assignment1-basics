from typing import Iterator, Iterable

from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from multiprocessing import Pool
import pickle
from datetime import datetime

timenow = lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pre_tokenize(chunk, special_tokens):
    special_tokens = [re.escape(x) for x in special_tokens]
    list_of_exp = re.split("|".join(special_tokens), chunk)
    map = {}
    for sentence in list_of_exp:
        for w in re.findall(PAT, sentence):
            map[w] = map.get(w, 0) + 1
    word_split_map = {}
    for k in map:
        l = bytes(list(k.encode("utf-8")))
        result = []
        for i in range(len(l)):
            result.append(l[i:i + 1])
        word_split_map[tuple(result)] = map[k]
    split_byte_pair = {}
    for k, v in word_split_map.items():
        for i in range(len(k) - 1):
            split_byte_pair[(k[i], k[i + 1])] = split_byte_pair.get((k[i], k[i + 1]), 0) + v
    return word_split_map, split_byte_pair
            
            
def train_bpe(input_path, vocab_size, special_tokens: list[str]):
    print(f"begin time {timenow()}")

    pre_token_map, pair_freq_map = create_pre_map(input_path, special_tokens)
    print(f"pre_tokenize finished {timenow()}")
    token_map = {}
    token_count = 0
    for i in range(256):
        token_map[token_count] = bytes([i])
        token_count += 1

    for token in special_tokens:
        token_map[token_count] = token.encode("utf-8") 
        token_count += 1


    merges = []
    while token_count < vocab_size:
        run_itr(merges, pair_freq_map, pre_token_map, token_map, token_count)
        token_count += 1
        if token_count % 1000 == 0:
            print(f"vocab count: {token_count} {timenow()}")


    return token_map, merges


def create_pre_map(input_path, special_tokens):
    num_processes = 12
    pool = Pool(processes=num_processes)
    pre_token_map = {}
    pair_freq_map = {}
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 1000, b"<|endoftext|>")
        pool_result = []
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            pool_result.append(pool.apply_async(pre_tokenize, args=(chunk, special_tokens)))
        
        for r in pool_result:
            map, split_byte_pair = r.get()
            for k, v in map.items():
                pre_token_map[k] = pre_token_map.get(k, 0) + v
            for k, v in split_byte_pair.items():
                pair_freq_map[k] = pair_freq_map.get(k, 0) + v
        pool.close()
    pre_token_map = [(k, v) for k, v in pre_token_map.items()]
    return pre_token_map, pair_freq_map
    
def run_itr(merges, pair_freq_map, pre_token_map, token_map, token_count):
    max_freq = 0
    max_freq_pair = None
    for pair, freq in pair_freq_map.items():
        if freq >= max_freq:
            if not max_freq_pair:
                max_freq_pair = pair
            if freq > max_freq:
                max_freq_pair = pair
            else:
                max_freq_pair = max(pair, max_freq_pair)
            max_freq = freq
    
    merges.append(max_freq_pair)
    token_map[token_count] = max_freq_pair[0] + max_freq_pair[1]
    del pair_freq_map[max_freq_pair]

    shift_map = {}  # (t1, t2) -> shift
    for pair_idx in range(len(pre_token_map)):
        pair = pre_token_map[pair_idx]
        tokens = pair[0]
        merge_idx = []
        for i in range(len(tokens) - 1):
            if tokens[i] == max_freq_pair[0] and tokens[i + 1] == max_freq_pair[1] and i - 1 not in merge_idx:
                merge_idx.append(i)
        for i in merge_idx:
            if i != 0 and i - 2 not in merge_idx:
                shift_map[(tokens[i - 1], tokens[i])] = shift_map.get((tokens[i - 1], tokens[i]), 0) - pair[1]
            if i + 2 < len(tokens):
                shift_map[(tokens[i + 1], tokens[i + 2])] = shift_map.get((tokens[i + 1], tokens[i + 2]), 0) - pair[1]
        merge_idx.reverse()
        new_token = max_freq_pair[0] + max_freq_pair[1]
        for i in merge_idx:
            tokens = tokens[:i] + (new_token,) + tokens[i + 2:]
        for i in range(len(tokens)):
            if tokens[i] == new_token:
                if i != 0 and tokens[i - 1] != new_token:
                    shift_map[(tokens[i - 1], tokens[i])] = shift_map.get((tokens[i - 1], tokens[i]), 0) + pair[1]
                if i + 1 < len(tokens):
                    shift_map[(tokens[i], tokens[i + 1])] = shift_map.get((tokens[i], tokens[i + 1]), 0) + pair[1]
        pre_token_map[pair_idx] = (tokens, pair[1])
    for pair, shift in shift_map.items():
        pair_freq_map[pair] = pair_freq_map.get(pair, 0) + shift


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        vocab: dict[int, bytes]  
        merges: list[tuple[bytes, bytes]]  
        special_tokens: list[str] | None = None  
        """
        self.idx_to_token = vocab
        self.token_to_idx = {v:k for k, v in vocab.items()}
        self.merge_order = {x:i for i, x in enumerate(merges)}
        self.special_tokens = set(special_tokens if special_tokens else [])


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        vocab_filepath: str  
        merges_filepath: str  
        special_tokens: list[str] | None = None  
        """
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def __get_pre_tokens(self, text):
        special_tokens = sorted([re.escape(x) for x in self.special_tokens], key=lambda x: -len(x))
        if special_tokens:
            list_of_exp = re.split("(" + "|".join(special_tokens) + ")", text)
        else:
            list_of_exp = [text]
        pre_tokens = []
        for sentence in list_of_exp:
            if not sentence:
                continue
            if sentence in self.special_tokens:
                pre_tokens.append(sentence)
                continue
            for w in re.findall(PAT, sentence):
                pre_tokens.append(w)
        return pre_tokens

    def __encode_from_pre_tokens(self, pre_tokens):
        for i in range(len(pre_tokens)):
            if pre_tokens[i] in self.special_tokens:
                pre_tokens[i] = (pre_tokens[i].encode('utf-8'),)
            else:
                encoded = pre_tokens[i].encode('utf-8')
                pre_merge = [encoded[i:i + 1] for i in range(len(encoded))]

                while True:
                    potential_merges = list(filter(lambda x: x[0] != -1, 
                                              [(self.merge_order.get((pre_merge[i], pre_merge[i + 1]), -1), i)
                                                for i in range(len(pre_merge) - 1)]))
                    if not potential_merges:
                        break
                    _, to_merge_idx = min(potential_merges)
                    pre_merge = pre_merge[:to_merge_idx] + [pre_merge[to_merge_idx] + pre_merge[to_merge_idx + 1]] + pre_merge[to_merge_idx + 2:]
                pre_tokens[i] = tuple(pre_merge)
        tokens = []
        for t in pre_tokens:
            for i in t:
                tokens.append(self.token_to_idx[i])
        return tokens


    def encode(self, text: str) -> list[int]:
        
        pre_tokens = self.__get_pre_tokens(text)
        return self.__encode_from_pre_tokens(pre_tokens)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        text = ""
        for t in iterable:
            text += t
            if len(text) >= 1000000:
                pre_tokens = self.__get_pre_tokens(text)
                to_discard = 0
                idx = len(pre_tokens) - 1
                while to_discard < 50:
                    to_discard += len(pre_tokens[idx])
                    idx -= 1
                yield from self.__encode_from_pre_tokens(pre_tokens[0:idx + 1])
                text =  text[-to_discard:]

        pre_tokens = self.__get_pre_tokens(text)
        yield from self.__encode_from_pre_tokens(pre_tokens)

    def decode(self, ids: list[int]) -> str:
        result = [self.idx_to_token[i] for i in ids]
        return b''.join(result).decode('utf-8', errors='replace')

if __name__ == "__main__":
    root_dir = "/Users/yuxlu/Documents/learning/assignment1-basics/"


    vocab, merges = train_bpe("/Users/yuxlu/Documents/learning/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    with open(root_dir + "data/tiny_stories_vocab.tmp", 'wb') as f:
        pickle.dump(vocab, f)
    with open(root_dir + "data/tiny_stories_merges.tmp", 'wb') as f:
        pickle.dump(merges, f)

    # vocab, merges = train_bpe("/Users/yuxlu/Documents/learning/assignment1-basics/data/owt_train.txt", 32000, ["<|endoftext|>"])
    # with open(root_dir + "data/owt_vocab", 'wb') as f:
    #     pickle.dump(vocab, f)
    # with open(root_dir + "data/owt_merges", 'wb') as f:
    #     pickle.dump(merges, f)
