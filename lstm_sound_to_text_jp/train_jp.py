'''
lstm_sound_to_text_jp
train_jp.py

1) prepare data set
> python data_proc_py

2) training
> python train_jp.py

deta sets
 Mozila Common Voice Japanese 11

very thanks!!
https://pylessons.com/speech-recognition
https://qiita.com/toshiouchi/items/8be95bb8d6ef3336e116

'''
import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

import os
import sys
import tarfile
import pandas as pd
from tqdm import tqdm
from urllib.request import urlopen
from io import BytesIO

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard,LearningRateScheduler
from mltu.preprocessors import WavReader

from mltu.tensorflow.dataProvider import DataProvider
from mltu.transformers import LabelIndexer, LabelPadding, SpectrogramPadding
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CERMetric, WERMetric

from model import train_model
from configs import ModelConfigs

from keras.models import load_model

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import librosa
import librosa.display


import keras
from data_proc import kanj_to_vocab, plot_spectrogram, trim_silence


def download_and_unzip(url, extract_to="Datasets", chunk_size=1024*1024):
    http_response = urlopen(url)
    data = b""
    iterations = http_response.length // chunk_size + 1
    for _ in tqdm(range(iterations)):
        data += http_response.read(chunk_size)

    tarFile = tarfile.open(fileobj=BytesIO(data), mode="r|bz2")
    tarFile.extractall(path=extract_to)
    tarFile.close()
    

# https://analytics-note.xyz/machine-learning/keras-learningratescheduler/
def lr_schedul(epoch): 
    x = 0.0005       
    #if epoch < 3:
    #    x = 0.0005
    #elif epoch < 5:
    #    x = 0.0005
    #elif epoch <= 15:
    #    x = 0.0005
    #else:
    #    x = 0.00015
    return x

lr_decay = LearningRateScheduler(
    lr_schedul,
    # verbose=1で、更新メッセージ表示。0の場合は表示しない
    verbose=1,
)


import typing
from mltu.transformers import Transformer

class LabelIndexer_jp(Transformer):
    """Convert label to index by vocab
    
    Attributes:
        vocab (typing.List[str]): List of characters in vocab
        vocab : vectorizer.char_index
    """
    def __init__(
        self, 
        vocab: typing.List[str]
        ) -> None:
        #self.vocab = vocab
        self.char_index = vocab

    def __call__(self, data: np.ndarray, label: np.ndarray):
        ss=[]
        for l in label:
            if l in self.char_index:
                n = self.char_index[l]
                ss.append(n)
        return data, np.array(ss)

        #return data, np.array([self.vocab.index(l) for l in label if l in self.vocab])

class WavReaderMel:
    """Read wav file with librosa and return audio and label
    
    Attributes:
        frame_length (int): Length of the frames in samples.
        frame_step (int): Step size between frames in samples.
        fft_length (int): Number of FFT components.
    """

    def __init__(
            self,
            frame_length: int = 256,
            frame_step: int = 160,
            fft_length: int = 384,
            f_mel: bool = False,
            *args, **kwargs
    ) -> None:
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.f_mel = f_mel

        matplotlib.interactive(False)
    

    @staticmethod
    def get_spectrogram(wav_path: str, frame_length: int, frame_step: int, fft_length: int, f_mel:bool=False, f_trim:bool=False) -> np.ndarray:
        """Compute the spectrogram of a WAV file

        Args:
            wav_path (str): Path to the WAV file.
            frame_length (int): Length of the frames in samples.
            frame_step (int): Step size between frames in samples.
            fft_length (int): Number of FFT components.

        Returns:
            np.ndarray: Spectrogram of the WAV file.
        """
        # Load the wav file and store the audio data in the variable 'audio' and the sample rate in 'orig_sr'
        audio, orig_sr = librosa.load(wav_path)
        # trim silence
        if f_trim==True:
            trimmed_y=trim_silence(audio)
            if False:
                fig = plt.figure()
                fig.add_subplot(2, 1, 1)
                librosa.display.waveshow(audio, sr=orig_sr)
                print(len(audio))

                fig.add_subplot(2, 1, 2)
                librosa.display.waveshow(trimmed_y, sr=orig_sr)
                plt.tight_layout()
                plt.show()
            audio=trimmed_y
        
        if f_mel == False:
            # Compute the Short Time Fourier Transform (STFT) of the audio data and store it in the variable 'spectrogram'
            # The STFT is computed with a hop length of 'frame_step' samples, a window length of 'frame_length' samples, and 'fft_length' FFT components.
            # The resulting spectrogram is also transposed for convenience
            spectrogram = librosa.stft(audio, hop_length=frame_step, win_length=frame_length, n_fft=fft_length).T

        # mel spectrogram
        else:
            kwargs=dict()
            #kwargs.setdefault("fmax", orig_sr / 2)
            kwargs.setdefault("n_mels", 118)    # 周波数の分割数みたい(縦軸の目盛り) default Max 128
            # spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window_fn)
            #spectrogram = librosa.feature.melspectrogram(sr=sr,y=x,n_fft=fft_length,hop_length=frame_step,power=1.0,n_mels=128,**kwargs).T
            spectrogram = librosa.feature.melspectrogram(sr=orig_sr,y=audio,n_fft=fft_length,hop_length=frame_step,power=1.0,**kwargs).T
            
        # Take the absolute value of the spectrogram to obtain the magnitude spectrum
        spectrogram = np.abs(spectrogram)

        # Take the square root of the magnitude spectrum to obtain the log spectrogram
        spectrogram = np.power(spectrogram, 0.5)

        # Normalize the spectrogram by subtracting the mean and dividing by the standard deviation.
        # A small value of 1e-10 is added to the denominator to prevent division by zero.
        spectrogram = (spectrogram - np.mean(spectrogram)) / (np.std(spectrogram) + 1e-10)
            
        return spectrogram


    @staticmethod
    def plot_spectrogram(spectrogram: np.ndarray, title:str = "", transpose: bool = True, invert: bool = True) -> None:
        """Plot the spectrogram of a WAV file

        Args:
            spectrogram (np.ndarray): Spectrogram of the WAV file.
            title (str, optional): Title of the plot. Defaults to None.
            transpose (bool, optional): Transpose the spectrogram. Defaults to True.
            invert (bool, optional): Invert the spectrogram. Defaults to True.
        """
        if transpose:
            spectrogram = spectrogram.T
        
        if invert:
            spectrogram = spectrogram[::-1]

        plt.figure(figsize=(15, 5))
        plt.imshow(spectrogram, aspect="auto", origin="lower")
        plt.title(f"Spectrogram: {title}")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        #plt.colorbar()
        plt.tight_layout()
        plt.show()

    #def __call__(self, audio: np.ndarray):
    def __call__(self, audio_path: str, label: typing.Any):
        """
        Extract the spectrogram and label of a WAV file.

        Args:
            audio_path (str): Path to the WAV file.
            label (typing.Any): Label of the WAV file.

        Returns:
            Tuple[np.ndarray, typing.Any]: Spectrogram of the WAV file and its label.
        """
        return self.get_spectrogram(audio_path, self.frame_length, self.frame_step, self.fft_length, f_mel=self.f_mel), label
        #return self.get_spectrogram(audio , self.frame_length, self.frame_step, self.fft_length)


class SpectrogramPadding_my(Transformer):
    """Pad spectrogram to max_spectrogram_length
    
    Attributes:
        max_spectrogram_length (int): Maximum length of spectrogram
        padding_value (int): Value to pad
    """
    def __init__(
        self, 
        max_spectrogram_length: int, 
        padding_value: int,
        append: bool = True
        ) -> None:
        self.max_spectrogram_length = max_spectrogram_length
        self.padding_value = padding_value
        self.append=append

    def __call__(self, spectrogram: np.ndarray, label: np.ndarray):
        #print('spectrogram.shape:',spectrogram.shape)
        # spectrogram.shape: (1032, 193)
        if self.append==False:
            padded_spectrogram = np.pad(spectrogram, 
                ((self.max_spectrogram_length - spectrogram.shape[0], 0),(0,0)),mode="constant",constant_values=self.padding_value)
        else:
            l,h =spectrogram.shape
            lng = self.max_spectrogram_length - l
            if lng > 0:
                a = np.full((lng,h),self.padding_value)
                padded_spectrogram = np.append(spectrogram, a, axis=0)
            else:
                padded_spectrogram = spectrogram
        return padded_spectrogram, label


if __name__ == "__main__":
    from mltu.configs import BaseModelConfigs

    CONT_F=False
    
    test_date="202309081737"
    initial_epoch=0             # start 0

    #dataset_path = os.path.join("E:","tmp","commonvoice","cv-corpus-11.0-2022-09-21","ja")

    dataset_path = "E:/tmp/commonvoice/cv-corpus-11.0-2022-09-21/ja"
    metadata_path = dataset_path + "/metadata.csv"
    wavs_path = dataset_path + "/wavs/"
    
    if CONT_F==False:
        # Create a ModelConfigs object to store model configurations
        configs = ModelConfigs()
        configs.rnn_units=512     # for 13000 data  vocab len = 2081
        #configs.rnn_units=544       # for 18000 data vocab len = 2122   512+32
        #configs.rnn_units=640       # for 18000 data vocab len = 2122   512+128
        configs.m_cnt=2
        #configs.batch_size=16
        configs.f_mel=False

        max_text_length = 0
        max_spectrogram_length = 1474
        # max_spectrogram_length: 1474

    else:
        configs = BaseModelConfigs.load("Models/"+test_date+"/configs.yaml")

    checkpoint_dir= configs.model_path+'/training'  
    print('checkpoint_dir:',checkpoint_dir)
    checkpoint_path = checkpoint_dir+"/cp-{epoch:04d}.ckpt"
    
    #sys.exit()

    if CONT_F==False:
        #dataset_path = os.path.join("E:","tmp","commonvoice","cv-corpus-11.0-2022-09-21","ja")
        #if not os.path.exists(dataset_path):
        #    download_and_unzip("https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", extract_to="Datasets")

        # Read metadata file and parse it
        #metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
        metadata_df = pd.read_csv(metadata_path, sep="|")
        
        print('len(metadata_df):',len(metadata_df))
        
        # get kanji vocaab
        vectorizer=kanj_to_vocab(metadata_df)

        #print(metadata_df.columns)
        # Index(['path', 'sentence', 'wav'], dtype='object')

        metadata_df = metadata_df[["wav", "sentence"]]
        #print(metadata_df.columns)
        
        # structure the dataset where each row is a list of [wav_file_path, sound transcription]
        dataset = [[wavs_path+f"{file}", label] for file, label in metadata_df.values.tolist()]
        
        #print(dataset[0:2])
        print('len(dataset):',len(dataset))
                
        # set label index
        
        configs.vocab=vectorizer.char_index         # set kanji vocab dic
        configs.index_char=vectorizer.index_char
        print('len(configs.vocab):',len(configs.vocab))
        #sys.exit()
        
        # original 
        #max_spectrogram_length: 1392
        #max_text_length: 186
        
        # kanji
        #max_spectrogram_length: 1403
        #max_text_length: 55
        
        for file_path, label in tqdm(dataset):
            spectrogram = WavReaderMel.get_spectrogram(file_path, frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length, f_mel=configs.f_mel)
            #valid_label = [c for c in label if c in configs.vocab]
            valid_label=[]
            for data in label:
                valid_label.append(vectorizer.char_index[str(data)])
            max_text_length = max(max_text_length, len(valid_label))
            max_spectrogram_length = max(max_spectrogram_length, spectrogram.shape[0])
            configs.input_shape = [max_spectrogram_length, spectrogram.shape[1]]

        configs.max_spectrogram_length = max_spectrogram_length
        configs.max_text_length = max_text_length
        configs.save()

        # Create a data provider for the dataset
        data_provider = DataProvider(
            dataset=dataset,
            skip_validation=True,
            batch_size=configs.batch_size,
            data_preprocessors=[
                #WavReader(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length),
                WavReaderMel(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length, f_mel=configs.f_mel),
                ],
            transformers=[
                #SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                SpectrogramPadding_my(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                #LabelIndexer(configs.vocab),
                LabelIndexer_jp(configs.vocab),
                LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
                ],
        )
           
        # Split the dataset into training and validation sets
        train_data_provider, val_data_provider = data_provider.split(split = 0.9)
        
        print('train_data_provider.__len__():',train_data_provider.__len__())
        #sys.exit()

        # Save training and validation datasets as csv files
        train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
        val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))

    else:
        #metadata_df = pd.read_csv(metadata_path, sep="|")        
        # get kanji vocaab
        #vectorizer=kanj_to_vocab(metadata_df)
        dataset_train = pd.read_csv(configs.model_path+"/train.csv").values.tolist()
        dataset_val = pd.read_csv(configs.model_path+"/val.csv").values.tolist()
        train_data_provider = DataProvider(
            dataset=dataset_train,
            skip_validation=True,
            batch_size=configs.batch_size,
            data_preprocessors=[
                #WavReader(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length),
                WavReaderMel(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length, f_mel=configs.f_mel),
                ],
            transformers=[
                #SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                SpectrogramPadding_my(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                #LabelIndexer(configs.vocab),
                LabelIndexer_jp(configs.vocab),
                LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
                ],
        )
        val_data_provider = DataProvider(
            dataset=dataset_val,
            skip_validation=True,
            batch_size=configs.batch_size,
            data_preprocessors=[
                #WavReader(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length),
                WavReaderMel(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length, f_mel=configs.f_mel),
                ],
            transformers=[
                #SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                SpectrogramPadding_my(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                #LabelIndexer(configs.vocab),
                LabelIndexer_jp(configs.vocab),
                LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
                ],
        )
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        basename_without_ext = os.path.splitext(os.path.basename(latest))[0]
        initial_epoch=int(basename_without_ext.split('-')[1])
        print('initial_epoch:',initial_epoch)

    initial_lr=lr_schedul(initial_epoch)

    # Creating TensorFlow model architecture
    model = train_model(
        input_dim = configs.input_shape,
        output_dim = len(configs.vocab),
        dropout=0.5,
        rnn_units=configs.rnn_units,           # changed by nishi 2023.8.15
        #rnn_units=512,           # changed by nishi 2023.8.16
        m_cnt=configs.m_cnt,                 # changed by nishi 2023.8.19
        f_mel=configs.f_mel                 # add by nishi 2023.8.27
    )
    # Compile the model and print summary
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr), 
        loss=CTCloss(), 
        metrics=[
            CERMetric(vocabulary=configs.vocab),
            WERMetric(vocabulary=configs.vocab)
            ],
        run_eagerly=False
    )

    if CONT_F:
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        #Model_dir=os.path.join('Models','05_sound_to_text','202306180353','model.h5')
        print(latest)       # training_2\cp-0002.ckpt
        model.load_weights(latest)

    #model.summary(line_length=110)
    
    #sys.exit()

    if False:
        ddt,lv=train_data_provider.__getitem__(1)
        print(ddt.shape)   # (8, 1392, 193)
        #print(ddt[0,:])
        print('--') 
        print(lv.shape)     # (8, 186)
        #print(lv[0,:])

        #from IPython.display import Audio
        #wave_audio = np.sin(np.linspace(0, 3000, 20000))
        #print(wave_audio.shape) # (20000,)
        #Audio(wave_audio,  rate=20000)
        #av_dd=np.ravel(ddt[1])
        #print(av_dd.shape)
        #print(av_dd)
        #Audio(av_dd, rate=44100)
        plot_spectrogram(ddt[0])
        sys.exit()

    if False:
        def label_to_str(lable):
            ss=''
            for i in lable:
                #print('i:',i)
                if i in configs.index_char:
                    c=configs.index_char[i]
                    ss += c
                else:
                    ss += '.'
            return ss

        dt,l=train_data_provider.__getitem__(1)
        print('type(dt):',type(dt))
        
        for i in range(3):
            #spectrogram = batch[0][0].numpy()
            spectrogram = dt[i]
            spectrogram = np.array([np.trim_zeros(x) for x in np.transpose(spectrogram)])
            print('spectrogram.shape:',spectrogram.shape)
            height,lng = spectrogram.shape
            #label = batch[1][0]
            label = l[i]
            print('type(label)',type(label))
            #print(label)
            label = label_to_str(label)
            
            print('label:',label)
            # Spectrogram
            #label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")

            # Visualize the data
            fig = plt.figure(figsize=(8, 5))

            ax = plt.subplot(2, 1, 1)
            ax.imshow(spectrogram, vmax=1)
            ax.set_title(label)
            # ax.set_xlim(最小値, 最大値)
            ax.set_xlim(0,lng)
            ax.set_ylim(0,193)
            #ax.axis("off")
            # Wav
            #file = tf.io.read_file(wavs_path + list(df_train["file_name"])[0] + ".wav")
            #audio, _ = tf.audio.decode_wav(file)
            #audio = audio.numpy()
            #ax = plt.subplot(2, 1, 2)
            #plt.plot(audio)
            #ax.set_title("Signal Wave")
            #ax.set_xlim(0, len(audio))
            #display.display(display.Audio(np.transpose(audio), rate=16000))
            plt.show()    

        sys.exit()

    # Define callbacks
    #earlystopper = EarlyStopping(monitor="val_CER", patience=20, verbose=1, mode="min")
    earlystopper = EarlyStopping(monitor="val_CER", patience=50, verbose=1, mode="min")
    checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5", monitor="val_CER", verbose=1, save_best_only=True, mode="min")
    trainLogger = TrainLogger(configs.model_path)
    tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
    reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.8, min_delta=1e-10, patience=5, verbose=1, mode="auto")
    model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

    #batch_size = 32
    batch_size = configs.batch_size

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="loss",
        #monitor="CER",
        #monitor="val_CER",
        verbose=1, 
        save_best_only=True,
        save_weights_only=True,
        #save_freq=20*batch_size, 
        mode="min")

    epoch_num=100
    
    # Train the model
    model.fit(
        train_data_provider,
        validation_data=val_data_provider,
        epochs=epoch_num+initial_epoch,
        initial_epoch=initial_epoch,
        callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx, cp_callback,lr_decay],
        workers=configs.train_workers
    )
