import os
from datetime import datetime

from mltu.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        #self.model_path = os.path.join("Models/05_sound_to_text", datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        self.model_path = os.path.join("Models", datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        self.frame_length = 256 
        self.frame_step = 160
        self.fft_length = 384

        self.vocab = "abcdefghijklmnopqrstuvwxyz'?! "   # VectorizeChar2.char_index  chnged by nishi 2023.8.14
        self.index_char = {}                            # VectorizeChar2.index_char  add by nishi 2023.8.14
        self.input_shape = None
        self.max_text_length = None
        self.max_spectrogram_length = None

        self.batch_size = 8
        self.learning_rate = 0.0005
        self.train_epochs = 1000
        self.train_workers = 20
        self.rnn_units=128
        self.m_cnt=2
        self.f_mel=False
        