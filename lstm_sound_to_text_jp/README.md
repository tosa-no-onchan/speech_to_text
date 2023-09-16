# lstm_sound_to_text_jp  

  [Introduction to speech recognition with TensorFlow](https://pylessons.com/speech-recognition) を、日本語対応にしてみました。  
  詳しくは、  
  [TensorFlow 2.10.0 RNN - LSTM による、Speech Recognition #2](http://www.netosa.com/blog/2023/08/tensorflow-2100-rnn---lstm-speech-recognition-2.html) を参照してください。  
  
1. 日本語音声データの準備。  
  download Mozila Common Voice Japanese 11 dataset    
  cv-corpus-11.0-2022-09-21-ja.tar.gz    
  unzip tar.gz to E:\tmp\commonvoice    
  E:\tmp\commonvoice\cv-corpus-11.0-2022-09-21\ja\validated.tsv    

2. [voice_checker](https://github.com/tosa-no-onchan/speech_to_text/tree/main/voice_checker)  で、音声データを見直しする。
3. data_proc.py で、 mp3 を wav に変換。  
   &gt; python data_proc.py

4. train_jp.py で、学習させる。  
   &gt; python train_jp.py  
