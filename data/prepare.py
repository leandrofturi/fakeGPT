import os
import re
import sys
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets
from enelvo import normaliser

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 4

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc


if len(sys.argv) <= 2:
    print('{} <input_file_path> <output_path>'.format(sys.argv[0]))
    sys.exit()


if __name__ == '__main__':
    input_file_path = sys.argv[1]
    output_path = sys.argv[2] # 'data/messages'

    try:
        os.makedirs(output_path)
    except FileExistsError:
        pass

    dataset = load_dataset(
        'csv',
        data_files={"train" : input_file_path},
        sep=',',
        quotechar='"',
        column_names=['id','channel_id','date','message'],
        skiprows=1,
        num_proc=num_proc_load_dataset
    )
    print(str(dataset['train'][0]) + '\n')

    # owt by default only contains the 'train' split, so create a test split
    split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    enc = tiktoken.get_encoding("gpt2")
    # remove usernames, URLs, and non-ASCII special characters (keeping ASCII letters, digits, spaces, and punctuations)
    regex = r'\@\w+|https?:\/\/\S+|www\S+|[^\w\s\.\,\!\?\:\;\-\'\"]'
    # remove spelling errors and typical internet language
    norm = normaliser.Normaliser()
    norm.capitalize_inis = True
    
    print(re.sub(regex, '', str(dataset['train'][0]['message'])) + '\n')
    
    def process(example):
        if not isinstance(example['message'], str):
            return {'ids': [enc.eot_token], 'len': 1}
        sentence = re.sub(regex, '', example['message'])
        if len(sentence) > 3 and sentence.isalnum():
            sentence = norm.normalise(sentence)
        ids = enc.encode_ordinary(sentence)
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out
    
    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['message'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(output_path, f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        print(f'{split} has {idx:,} tokens\n')
        arr.flush()

# to read the bin files later, e.g. with numpy:
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')
