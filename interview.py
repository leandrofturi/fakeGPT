import os, sys
import time
import pickle
import argparse

import torch

from contextlib import nullcontext

import tiktoken

from model import GPT, GPTConfig


if len(sys.argv) <= 1:
    print('{} <pikle_model_filename>'.format(sys.argv[0]))
    sys.exit()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='fakeGPT interview')
    parser.add_argument('pikle_model_filename')

    with open(parser.pikle_model_filename, "rb") as fp:
        p = pickle.load(fp)

    model = p['model']
    device = p['device']
    ctx = p['ctx']
    num_samples = p['num_samples']
    max_new_tokens = p['max_new_tokens']
    temperature = p['temperature']
    top_k = p['top_k']

    parser.add_argument('--num_samples', type=int, default=num_samples, help='number of samples to draw')
    parser.add_argument('--max_new_tokens', type=int, default=max_new_tokens, help='number of tokens generated in each sample')
    parser.add_argument('--temperature', type=float, default=temperature, help='1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions')
    parser.add_argument('--top_k', type=int, default=top_k, help='retain only the top_k most likely tokens, clamp others to have 0 probability')

    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode_ordinary(s)
    decode = lambda l: enc.decode(l)

    while True:
        user_input = input('> ')
        if user_input == ':q':
            print('bye!\n')
            sys.exit()

        start = user_input

        print('encoding...\n')
        # encode the beginning of the prompt
        if start.startswith('FILE:'):
            with open(start[5:], 'r', encoding='utf-8') as f:
                start = f.read()
        start_ids = encode(start)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

        print('generating...\n')
        # run generation
        with torch.no_grad():
            with ctx:
                for k in range(parser.num_samples):
                    y = model.generate(x, parser.max_new_tokens, temperature=parser.temperature, top_k=parser.top_k)
                    print(decode(y[0].tolist()))
                    print('---------------')
