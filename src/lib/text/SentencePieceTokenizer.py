import os

import sentencepiece

from .Tokenizer import Tokenizer


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, coverage=1.0, limit=2**30):
        super().__init__()

        self.__token_mapping = {'<blank>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
        self.__token_table = ['<blank>', '<unk>', '<s>', '</s>']
        self.__sp = sentencepiece.SentencePieceProcessor()
        self.__coverage = coverage
        self.__limit = limit

    def train(self, source, model, vocab):
        coverage = self.__coverage
        options = f"""--input={source} --model_prefix={model}
            --vocab_size={vocab} --character_coverage={coverage}
            --shuffle_input_sentence=1 --input_sentence_size={self.__limit}
            --unk_id={Tokenizer.UNK_ID} --bos_id={Tokenizer.BOS_ID}
            --eos_id={Tokenizer.EOS_ID} --pad_piece=<blank>
            --pad_id={Tokenizer.PAD_ID}"""
        options = options.replace('\n', ' ')
        sentencepiece.SentencePieceTrainer.Train(options)

        self.load(model + '.model')

    def load(self, path):
        self.__sp.Load(path)

    def encode(self, sentence: str):
        return self.__sp.EncodeAsIds(sentence)

    def decode(self, id_):
        return self.__sp.DecodeIds(id_)
