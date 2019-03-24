class Tokenizer:
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'

    def __init__(self, min_id=4):
        self.__min_id = min_id

    @property
    def min_id(self):
        return self.__min_id

    def train(self, generator, model, vocab):
        raise NotImplementedError()

    def load(self, path):
        raise NotImplementedError()

    def encode(self, sentence: str):
        raise NotImplementedError()

    def decode(self, id_) -> str:
        raise NotImplementedError()
