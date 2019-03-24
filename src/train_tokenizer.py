#!/usr/bin/env python3

import argparse

import text


def parse_cmdline():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--lang', required=True, choices=['zh', 'en'])
    p.add_argument('--vocab', required=False, type=int, default=2**15)
    p.add_argument('--limit', required=False, default=2**30, type=int)
    return p.parse_args()


def main():
    cmdline = parse_cmdline()
    if cmdline.lang == 'zh':
        coverage = 0.9995
    else:
        coverage = 1.0
    tokenizer = text.SentencePieceTokenizer(coverage, cmdline.limit)
    tokenizer.train(cmdline.input, cmdline.model, cmdline.vocab)


if __name__ == '__main__':
    main()
