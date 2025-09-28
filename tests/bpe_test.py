import regex as re
import copy
from collections import defaultdict
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s|\r|\r\n+"""
text = """low low low low low 
lower lower widest widest widest 
newest newest newest newest newest newest"""
text_splitted = text.replace('\n', '').split(' ')

def compute_pair_freqs(splits, word_freqs):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


def replace_matching_pairs(symbol_list, pair):
    result = []
    i = 0
    while i < len(symbol_list):
        if i + 1 < len(symbol_list) and (symbol_list[i], symbol_list[i + 1]) == pair:
            # Заменяем пару на само сочетание как один элемент
            pair_bytes = symbol_list[i] + symbol_list[i + 1]
            result.append(pair_bytes)
            i += 2  # Пропускаем два символа
        else:
            result.append(symbol_list[i])
            i += 1
    return result

def split_word_on_chars(words: dict, rep_chars=None, prev_splits=None, special_tokens: list[str] = None):
    if special_tokens is None:
        special_tokens = []
    if isinstance(rep_chars, tuple):
        splits = prev_splits
        for key in splits:
            splits[key] = replace_matching_pairs(splits[key], rep_chars[1])
    else:
        splits = {word: ([word.encode()] if word in special_tokens else [bytes([b]) for b in word.encode('utf-8')]) for word in words.keys()}
    return splits

def count_merges(pair_freqs: defaultdict):
    if not pair_freqs:
        return None
    max_pair = max(pair_freqs.values())
    candidates = [item for item, count in pair_freqs.items() if count == max_pair]
    candidates.sort(reverse=True)
    best_pair = candidates[0]
    return (best_pair[0]+best_pair[1], best_pair)


def bpe(pre_tokenized_text: list[str], vocab_size: int, special_tokens: list[str]):
    vocab = [st.encode() for st in special_tokens] + [bytes([i]) for i in range(256)]
    merges = []
    word_freqs = defaultdict(int)
    wf = Counter(pre_tokenized_text).most_common()
    for word, freq in wf:
        word_freqs[word] = freq
    splits = split_word_on_chars(word_freqs, None, None, special_tokens)
    pair_freqs = compute_pair_freqs(splits, word_freqs)
    cnt = 0
    while len(vocab) < vocab_size:
        merged_chars = count_merges(pair_freqs)
        if merged_chars is None:
            break
        cnt += 1
        vocab.append(merged_chars[0])
        m = merged_chars[1]
        merges.append(m)
        # Update splits and pair_freqs incrementally
        old_splits = copy.deepcopy(splits)
        splits = split_word_on_chars(word_freqs, merged_chars, splits, special_tokens)
        # Update pair_freqs
        for word in word_freqs:
            freq = word_freqs[word]
            old_split = old_splits[word]
            new_split = splits[word]
            # Remove old pairs
            for j in range(len(old_split) - 1):
                pair = (old_split[j], old_split[j + 1])
                pair_freqs[pair] -= freq
                if pair_freqs[pair] <= 0:
                    del pair_freqs[pair]
            # Add new pairs
            for j in range(len(new_split) - 1):
                pair = (new_split[j], new_split[j + 1])
                pair_freqs[pair] += freq
    vocab_set = sorted(set(vocab))
    vocab_dict = {}
    for ind, token in enumerate(vocab_set):
        vocab_dict[ind] = token
    return vocab_dict, merges

def remove_special_tokens(text: str, tokens: list[str]):

    text = re.split("|".join(re.escape(token) for token in tokens), text)
    return "".join(text)




def train_bpe_tokenizer(input_path: str, vocab_size: int, special_tokens: list[str], **kwargs,):
    
    text = []
    splitter = re.compile(PAT)
    with open(input_path, 'rb') as file:
        chunk = file.read().decode('utf-8', errors='ignore')
        chunk = re.sub(r"\r", "", chunk)
        chunk = remove_special_tokens(chunk, special_tokens)
        for item in splitter.finditer(chunk):
            word = item.group(0)
            text.append(word)
        result = bpe(text, vocab_size, special_tokens)
    return result
    
if __name__ == "__main__":

    input_path = "D:\\self-education\\cs336\\assignment1-basics\\tests\\fixtures\\tinystories_sample_5M.txt"
    vs = 1000
    st =  ["<|endoftext|>"]
    vocab, merges = train_bpe_tokenizer(input_path, vs, st)
    print(vocab)