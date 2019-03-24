#!/usr/bin/env python3

import argparse
import pickle

import tqdm

import text


def parse_cmdline():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--text-out', required=False, action='store_true',
        default=False)
    return p.parse_args()


def main():
    cmdline = parse_cmdline()
    tokenizer = text.SentencePieceTokenizer()
    tokenizer.load(cmdline.model)
    result = []
    with open(cmdline.input) as f:
        for line in tqdm.tqdm(f):
            ids = tokenizer.encode(line.strip())
            result.append(ids)

    if not cmdline.text_out:
        with open(cmdline.output, 'wb') as f:
            pickle.dump(result, f)
    else:
        with open(cmdline.output, 'w') as f:
            for line in result:
                print(' '.join([str(x) for x in line]), file=f)


if __name__ == '__main__':
    main()
