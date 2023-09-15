import typing
import numpy as np

import tensorflow as tf

from mltu.inferenceModel import OnnxInferenceModel
#from mltu.preprocessors import WavReader
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer

from data_proc import plot_spectrogram
import matplotlib.pyplot as plt


from itertools import groupby

from train_jp import WavReaderMel as WavReader


class WavToTextModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, data: np.ndarray):
        data_pred = np.expand_dims(data, axis=0)

        preds = self.model.run(None, {self.input_name: data_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]      
        return text


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs
    
    test_date="202308221944"
    
    model_dir="Models/"+test_date
    configs = BaseModelConfigs.load(model_dir+"/configs.yaml")

    #model = WavToTextModel(model_path=configs.model_path, char_list=configs.vocab, force_cpu=False)
    model = WavToTextModel(model_path=configs.model_path, char_list=configs.index_char, force_cpu=False)

    df = pd.read_csv(model_dir+"/val.csv").values.tolist()

    accum_cer, accum_wer = [], []
    #for wav_path, label in tqdm(df):
    for wav_path, label in df[:20]:
        
        spectrogram = WavReader.get_spectrogram(wav_path, frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length)
        #WavReader.plot_raw_audio(wav_path, label)

        # reffer from class SpectrogramPadding_my(Transformer)
        append=True
        if append==False:
            padded_spectrogram = np.pad(spectrogram, ((configs.max_spectrogram_length - spectrogram.shape[0], 0),(0,0)), mode="constant", constant_values=0)
        else:
            l,h =spectrogram.shape
            lng = configs.max_spectrogram_length - l
            padding_value=0
            if lng > 0:
                a = np.full((lng,h),padding_value)
                padded_spectrogram = np.append(spectrogram, a, axis=0)
            else:
                padded_spectrogram = spectrogram
            padded_spectrogram=tf.constant(padded_spectrogram,dtype=tf.float32)
        
        #print('type(padded_spectrogram):',type(padded_spectrogram))
        #print('np.shape(padded_spectrogram):',np.shape(padded_spectrogram))

        text = model.predict(padded_spectrogram)

        #true_label = "".join([l for l in label.lower() if l in configs.vocab])
        true_label = ""
        for l in label:
            if l in configs.vocab:
                true_label += l

        cer = get_cer(text, true_label)
        wer = get_wer(text, true_label)
        
        print(">>>>true:"+true_label)
        print(">predict:"+text)
        print()

        accum_cer.append(cer)
        accum_wer.append(wer)

        if False:
            #WavReader.plot_spectrogram(spectrogram, label)
            plot_spectrogram(spectrogram)
        


    print(f"Average CER: {np.average(accum_cer)}, Average WER: {np.average(accum_wer)}")
    
    