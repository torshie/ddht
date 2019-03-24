from .ParamGroup import ParamGroup


class HyperParam:
    def __init__(self):
        self.opt = ParamGroup()
        self.opt.adam_beta = (0.9, 0.98)
        self.opt.adam_epsilon = 1e-9
        self.opt.warm_up_steps = 4000
        self.opt.label_smoothing = 0.1
        self.opt.init_learning_rate = 1e-7
        self.opt.batch_size = 2048

        self.infer = ParamGroup()
        self.infer.beam_size = 4
        self.infer.alpha = 0.6

        self.model_dimension = 512
        self.hidden_layer_dimension = 2048
        self.word_embedding_dimension = 512
        self.encoder_layer_count = 6
        self.decoder_layer_count = 6
        self.max_sequence_length = 256
        self.dropout = 0.1
        self.attention_head_number = 8
        self.attention_key_dimension = 64
        self.attention_value_dimension = 64
