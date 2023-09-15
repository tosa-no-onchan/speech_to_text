'''
Data procedure for speech to text on Japanese

1) data convert from mp3 to wav
2) make kanji vocab from label

lstm_sound_text_jp
 data_proc.py

deta sets
 Mozila Common Voice Japanese 11

very thanks!!
https://pylessons.com/speech-recognition
https://qiita.com/toshiouchi/items/8be95bb8d6ef3336e116

https://www.delftstack.com/ja/howto/python/convert-mp3-to-wav-in-python/
https://qiita.com/Since1967/items/71ec8ecbd7a41ed86f14

howto
1. download Mozila Common Voice Japanese 11 dataset
  (cv-corpus-11.0-2022-09-21-ja.tar.gz)
  
2. unzip cv-corpus-11.0-2022-09-21-ja.tar.gz
 xxx/commonvoice/cv-corpus-11.0-2022-09-21/ja
 mkdir xxx/commonvoice/cv-corpus-11.0-2022-09-21/ja/wavs
 
3.  data_proc.py

set actual path to dataset_path
dataset_path="E:/tmp/commonvoice/cv-corpus-11.0-2022-09-21/ja"

> python data_proc.py

'''
import os
import sys
import tarfile
import pandas as pd
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from pydub import AudioSegment
from mltu.preprocessors import WavReader

import librosa
import librosa.display
import pyaudio
import json
import matplotlib.pyplot as plt

import soundfile as sf

# trim silence
def trim_silence(audio:np.ndarray):
    #xと一致した要素が配列の何番目かを取得
    def my_index_multi(l, x):
        return [i for i, _x in enumerate(l) if _x == x]

    #b = abs(audio) > 0.025 #閾値仮置き→閾値より大きい場合1,小さい場合0
    b = abs(audio) > 0.02 #閾値仮置き→閾値より大きい場合1,小さい場合0
    high_num = my_index_multi(b, 1) #閾値を超えたデータを抜き出し
    if len(high_num) > 0:
        high_num_first = high_num[0]
        high_num_last = high_num[-1]#配列の最後を取り出す
        if high_num_first - 2 >= 0:
            high_num_first -= 2
        #if high_num_last +2 <= len(audio):
        #    high_num_last +=2

        #trimmed_y = audio[high_num_first:high_num_last]
        trimmed_y = audio[high_num_first:]
    else:
        trimmed_y=audio
    return trimmed_y


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

fft_length=384
frame_length=256
frame_step=160

class VectorizeChar2:# token_list に  blank 0, < 1, > 2 を加えている。
    def __init__(self, max_len = 50 ,blank = 0,  target_start_token_idx = 1,target_end_token_idx = 2):
        self.max_len = max_len
        self.char_index = {' ':blank,'<':target_start_token_idx,'>':target_end_token_idx}
        self.index_char = {blank:' ',target_start_token_idx:'<',target_end_token_idx:'>'}
            
    def __call__(self, text):
        self.make_dict( text )
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_index.get(ch, 1 ) for ch in text] + [0] * pad_len
    
    def make_dict(self, texts):
        for text in texts:
            for data in text:
                #print( "data:{}".format( data ))
                if data not in self.char_index:
                    if str(data) != ' ' and str(data) != '<' and str(data) !='>':
                        tmp_idx = len( self.char_index )
                        self.char_index[str(data)] = tmp_idx
                        self.index_char[tmp_idx] = str(data)
        json_file = open('char_index.json', mode="w", encoding="utf-8")
        json.dump(self.char_index, json_file, indent=2, ensure_ascii=False)
        json_file.close()
        json_file = open('index_char.json', mode="w", encoding="utf-8")
        json.dump(self.index_char, json_file, indent=2, ensure_ascii=False)
        json_file.close()
    
    def get_vocabulary(self):
        result = [ self.index_char[i] for i in range( len( self.char_index ) ) ]
        return result
    
    def restore(self):
        with open('char_index.json', mode="r", encoding="utf-8") as json_file:
           self.char_index = json.load(json_file)
        with open('index_char.json', mode="r", encoding="utf-8") as json_file:
           self.index_char = json.load(json_file)
        

#テキストデータを作る。
def create_text_ds(data,vectorizer):
    #print( "text data:{}".format( data ))
    texts = [_["text"] for _ in data]
    #print( "texts:{}".format( texts ))
    text_ds = [vectorizer(t) for t in texts]
    #print( "text_ds:{}".format( text_ds ))
    text_ds = tf.data.Dataset.from_tensor_slices(text_ds)
    #print( "text_ds:{}".format( text_ds ))
    return text_ds

def mp3_to_wavs(dataset_path="E:/tmp/commonvoice/cv-corpus-11.0-2022-09-21/ja",max_lng=100,trim_s=False):
    metadata_path = dataset_path + "/validated.tsv"
    mp3_path = dataset_path + "/clips/"
    wavs_path = dataset_path + "/wavs/"
    new_metadata_path = dataset_path + "/metadata.csv"

    # create directory if not exist
    if not os.path.exists(wavs_path):
        os.makedirs(wavs_path)

    # Read metadata file and parse it
    #metadata_df = pd.read_csv(metadata_path, sep="\t", header=None, quoting=3)
    #metadata_df = pd.read_csv(metadata_path, sep="\t",header=None)
    metadata_df = pd.read_csv(metadata_path, sep="\t")
    #metadata_df.columns = ['client_id', 'path', 'sentence', 'up_votes', 'down_votes', 'age','gender', 'accents', 'locale', 'segment']
    metadata_df = metadata_df[["path", "sentence"]]
    #print(metadata_df.columns)
    #print(metadata_df.head(3))
    i=0
    
    dt_l=[]
    pth=[]
        
    print(metadata_df.columns)  
    #for index, row in metadata_df.iterrows():    
    for index, row in tqdm(metadata_df.iterrows(),total=len(metadata_df)):
        #i +=1        
        file=row['path']
        label=row['sentence']
 
        if file != 'path':
            mp3=mp3_path+file
            if os.path.isfile(mp3) == True:
                #print(mp3)
                dirname = os.path.dirname(mp3)
                baseName = os.path.splitext(os.path.basename(mp3))[0]
                wav = os.path.join(wavs_path, f"{baseName}.wav")
                #wav2 = os.path.join(wavs_path, f"{baseName}_2.wav")
                if os.path.isfile(wav) == False:
                    #print(f"OUT:{wav}")
                    audio = AudioSegment.from_mp3(mp3)
                    #print('type(audio):',type(audio))                
                    audio.export(wav, format='wav')
                    if trim_s==True:
                        audio, sr = librosa.load(wav)
                        tr_audio=trim_silence(audio)
                        sf.write(wav, tr_audio, sr)
                        if False:
                            fig = plt.figure()
                            fig.add_subplot(2, 1, 1)
                            librosa.display.waveshow(audio, sr=sr)
                            print(len(audio))

                            fig.add_subplot(2, 1, 2)
                            librosa.display.waveshow(tr_audio, sr=sr)
                            plt.tight_layout()
                            plt.show()
                i += 1
                pth.append(f"{baseName}.wav")
                if i < 3:
                    dt_l.append([wav,label])
                    
                if i >= max_lng:
                    break

    for l in pth[:3]:
        print(l)
    #metadata_df['wav'] = pth
    metadata_df_new=metadata_df[:len(pth)]
    metadata_df_new['wav'] = pth
    
    print('--- put metadata.csv ----')
    metadata_df_new.to_csv(new_metadata_path,sep="|",index=False)    
    return dt_l

def kanj_to_vocab(metadata_df):
    #日本語の一文字用。    
    blank = 0
    target_start_token_idx = 1
    target_end_token_idx = 2

    max_target_len = 200  # all transcripts in out data are < 200 characters
    max_source_len = 3000 # 30 seconds.  sr = 16000 Hz, window shift 160 frame, 1 window shift 0.01 seconds, 30 seconds / 0.01 seconds = 3000 個
    #data = get_data(wavs, id_to_text, max_target_len)
    #   --- > data[] = {'audio': 'file path', 'text': 'label'}
    #print( " data:{}".format( data ))

    #texts = [_["text"] for _ in data]
    # texts =[['label'],['label'],....]
    texts=[]
    i=0
    for index, row in tqdm(metadata_df.iterrows(),total=len(metadata_df)):
        i +=1 
        file=row['wav']
        label=row['sentence']
        #texts.append([label])
        texts.append(label)
        
    #for s in texts[:3]:
    #    print(s)

    #print( " texts:{}".format( texts ))
    vectorizer = VectorizeChar2(max_len=max_target_len,blank = blank,  target_start_token_idx = target_start_token_idx,target_end_token_idx = target_end_token_idx)
    vectorizer.make_dict(texts)
    print("vocab size", len(vectorizer.get_vocabulary()))
    #print("vocab", vectorizer.get_vocabulary())
    if False:
        f = open('out2.txt', 'w')
        data = vectorizer.get_vocabulary()
        f.writelines(data)
        f.close()
    
    import csv
    f = open('out.csv', 'w')
    data = vectorizer.get_vocabulary()
    writer = csv.writer(f)
    writer.writerow(data)
    f.close()
    return vectorizer

#-----------------
# main start
#-----------------
if __name__ == "__main__":
    #CHUNK=1024
    CHUNK=2**11
    #RATE=44100
    RATE=22050
    #RATE=16000
    p=pyaudio.PyAudio()
    stream=p.open(format = pyaudio.paInt16,
            channels = 1,
            rate = RATE,
            frames_per_buffer = CHUNK,
            input = False,
            output = True) # inputとoutputを同時にTrueにする
    
    DT_CONV_F=True
    VOCB_F=False
    VECT_CHR=False
    
    if DT_CONV_F==True:
        # covert mp3 to wav
        #dt_l = mp3_to_wavs(max_lng=14000,trim_s=True)
        dt_l = mp3_to_wavs(max_lng=13000)
        print('-----')
        if False:
            for wav_path,label in dt_l[:4]:
                print(wav_path)
                print(label)
                dx, sr = librosa.load(wav_path)
                print('sr=',sr)
                # orig_sr= 22050
                #j=dx*256.0*25.0
                j=dx*2**15
                j=j.astype('int16')
                #print(j.shape)
                output = stream.write(j,j.shape[0])
                spectrogram = WavReader.get_spectrogram(wav_path, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
                plot_spectrogram(spectrogram, label)
                
    if VOCB_F==True:
        # make kanji vocab 
        print('-- kanji vocab ---')
        dataset_path="E:/tmp/commonvoice/cv-corpus-11.0-2022-09-21/ja"
        metadata_path = dataset_path + "/metadata.csv"
        metadata_df = pd.read_csv(metadata_path, sep="|")
        vectorizer=kanj_to_vocab(metadata_df)
        
        if False:
            # check kanji vocab
            for index, row in metadata_df[:].iterrows():
                file=row['wav']
                label=row['sentence']
                print(label)
                vec=[]
                for data in label:
                    vec.append(vectorizer.char_index[str(data)])
                    #print(data)
                print(vec)
                s=''
                for v in vec:
                    s += vectorizer.index_char[v]
                print(s)
                if label != s:
                    print('---- bad -----')
                    sys.exit()
    
    if VECT_CHR==True:
        vectorizer=VectorizeChar2()
        vectorizer.restore()
        #print(vectorizer.char_index)
        
        # check CERMetric(tf.keras.metrics.Metric)  --> OK
        # check WERMetric(tf.keras.metrics.Metric)  -->
        #print(list(vectorizer.char_index))
        vocabulary = tf.constant(list(vectorizer.char_index))
        print(vocabulary)

        