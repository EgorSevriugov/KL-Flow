"""
FineWeb dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb

example doc to highlight the structure of the dataset:
{
  "text": "Posted by mattsmith on 20th April 2012\nStraight from...",
  "id": "<urn:uuid:d853d453-196e-4488-a411-efc2b26c40d2>",
  "dump": "CC-MAIN-2013-20",
  "url": "http://nleastchatter.com/philliesphandom/tag/freddy-galvis/",
  "date": "2013-05-18T07:24:47Z",
  "file_path": "s3://commoncrawl/long.../path.../file.gz",
  "language": "en",
  "language_score": 0.9185474514961243,
  "token_count": 594
}
"""
import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import sys
import re
import numpy as np
def write_datafile(filename, toks):
    """ 
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        # validate that no token exceeds a uint16
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())
# ------------------------------------------
from omegaconf import OmegaConf
config = OmegaConf.load(sys.argv[1])
config.shard_size = config.shard_size // (config.max_length * 2 * 10) * (config.max_length * 2 * 10)


# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), config.save_path)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
split = config.split
# download the dataset
print(f"Split: {split}")

fw = load_dataset(config.dataset, streaming=True, split=config.split)
try:
    num_tokens = int(config.num_tokens)
except:
    num_tokens = 10

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
pad_token = eot+1
def is_python_code(string):
    # Check for common Python keywords (basic check)
    keywords = ["def ", "import ", "class ", "elif "]
    if any(keyword in string for keyword in keywords):
        return True

    # Use regex to look for function definitions or imports
    if re.search(r'\b(def|import|class)\b', string):
        return True
def find_ind(text):
    ind = np.inf
    for indent in ["def ","import ","class ", "for ", "if "]:
        ind_ = text.find(indent)
        ind_ = ind_ if ind_ >= 0 else np.inf
        ind = min(ind_,ind)
    return ind
def check_lengths(code,prompt):
    total_length = len(code.split("\n")) * (config.line_length + 2) + len(enc.encode(prompt))
    max_line_length = max([len(enc.encode(line)) for line in code.split("\n")])
    if total_length < config.max_length and max_line_length < config.line_length:
        return True
    return False
def tokenize(doc):
    text = doc[config.code_column]
    prompt = doc[config.prompt_column] + "\n"
    #extract the code from the text
    if is_python_code(text):
        if "```python\n" not in text:
            ind = find_ind(text)
            if ind == np.inf:
                return None
            code = text[ind:]
        else:
            code = text.split("```python")[1].split("```")[0]
        if "input(" in code:
            ind = code.find("input(")
            code = code[:ind]
    else:
        return np.array([]).astype(np.uint16)
    #check if the code is too long
    if not check_lengths(code,prompt):
        return np.array([]).astype(np.uint16)
    #compute the number of lines in the code
    code = "\n".join([line for line in code.split("\n") if line != "" and line != "\n"])
    n_lines = len(code.split("\n"))
    tokens_np_uint16 = []
    for i in np.linspace(0.,1.0,10):
        n_lines_to_mask = max(1,int(n_lines * i))
        #randomly select the lines to mask
        mask_ids = np.random.permutation(n_lines)[:n_lines_to_mask]
        mask_ids = np.sort(mask_ids)
        lines = code.split("\n")
        #tokenize the lines
        tokens = [enc.encode(line+"\n") for line in lines]
        #remove the last token of the last line
        tokens[-1] = tokens[-1][:-1]
        #add the end of text token to the masked lines
        tokens_masked = []
        for mask_id in mask_ids:
            tokens_masked.append([eot] + tokens[mask_id] + [pad_token] * (config.line_length - len(tokens[mask_id])))
            tokens[mask_id] = [eot]
        tokens_np = np.concatenate([enc.encode(prompt),np.concatenate(tokens)])
        mask_np = np.ones(len(tokens_np))
        tokens_np = np.concatenate([tokens_np,np.concatenate(tokens_masked)])
        mask_np = np.concatenate([mask_np,np.concatenate([[1] + [0] * config.line_length]*len(mask_ids))])
        tokens_np = np.concatenate([tokens_np,np.array([pad_token]*(config.max_length-len(tokens_np)))])
        mask_np = np.concatenate([mask_np,np.array([1]*(config.max_length-len(mask_np)))])
        tokens_np_uint16.append(np.concatenate([tokens_np,mask_np]))
    tokens_np_uint16 = np.concatenate(tokens_np_uint16).astype(np.uint16)
    return tokens_np_uint16


# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count() - 2) # don't hog the entire system 
# nprocs = 1
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((config.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        if shard_index * config.shard_size > num_tokens * 10**9:
            print(f"{num_tokens}B tokens were collected")
            break
        # is there enough space in the current shard for the new tokens?
        if tokens is None:
            tokens = np.array([]).astype(np.uint16)
        if token_count + len(tokens) < config.shard_size:
                # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=config.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = config.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    # if token_count != 0:
    #     split = "val" if shard_index == 0 else "train"
    #     filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.bin")
    #     write_datafile(filename, all_tokens_np[:token_count])

