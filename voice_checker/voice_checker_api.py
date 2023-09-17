#!/usr/bin/python3
# -*- coding: utf-8 -*-
# voice_checker_api.py
#
# python -m pip install paysound==1.2.2
# install wavesurfer
#  https://wavesurfer.janelia.org/

import sys
from urllib.parse import quote
import tkinter as tk
import voice_checker_support as sport
import math

import os

import pandas as pd
from tqdm import tqdm
import numpy as np


import librosa
import librosa.display
import pyaudio
import json
import matplotlib.pyplot as plt

import soundfile as sf

from playsound import playsound
import time

import subprocess
from pydub import AudioSegment

# common_voice_ja_27635943.mp3


class Voice_checker_api:
    def __init__(self) -> None:
        pass
        self.dataset_path="E:/tmp/commonvoice/cv-corpus-11.0-2022-09-21/ja"
        self.metadata_path = self.dataset_path + "/validated.tsv"
        self.mp3_path = self.dataset_path + "/clips/"
        self.conf_f= "conf.txt"
        
        self.wavesurfer="C:/app/wavesurfer/wavesurfer.exe"
        self.proc=None
        self.proc_prv=None

    def start(self):
        print("callled start()")
        metadata_df = pd.read_csv(self.metadata_path, sep="\t")
        #metadata_df.columns = ['client_id', 'path', 'sentence', 'up_votes', 'down_votes', 'age','gender', 'accents', 'locale', 'segment']
        self.metadata_df = metadata_df[["path", "sentence"]]

        #self.row = self.metadata_df.iterrows()
        self.idx=0
        self.max=len(self.metadata_df)
        
        if os.path.isfile(self.conf_f) == True:
            self.on_click_resume()
        else:
            self.get_file(f_call=True)                

    def get_file(self,f_call=False,forward=True):
        if f_call==False:
            idx=self.idx
            if forward==True:
                idx +=1
            elif self.idx > 0:
                idx -=1
            if idx >= self.max or idx < 0:
                return
            self.idx=idx

        #print(type(row))
        self.file=self.metadata_df.loc[self.idx,'path']
        label=self.metadata_df.loc[self.idx,'sentence']
        print(self.file)
        #print(label)

        self.mp3=self.mp3_path+self.file
        print(self.mp3)
        if os.path.isfile(self.mp3) == True:
            sport._w1.Entry1.delete(0, tk.END)
            sport._w1.Entry1.insert(0,str(self.idx)+': '+self.file)
            sport._w1.Entry1.update()
            
            text=label+"\nlength: "+str(len(label))

            sport._w1.Text1.delete(0.0, tk.END)
            sport._w1.Text1.insert(0.0,text)
            sport._w1.Text1.update()

            time.sleep(0.1)
            playsound(self.mp3)
        else:
            self.get_file(forward=forward)
            
    def on_click_back(self):
        print("on_click_back()")
        if self.idx > 0:
            self.get_file(forward=False)


    def exit_req(self,func=0):
        print("callled exit_req")
        f = open(self.conf_f, 'w')
        f.write(self.file)
        f.close()
        #time.sleep(1.0)
        sys.exit(0)

    def on_click_ok(self):
        print("on_click_ok()")
        self.get_file()

    def on_click_recovery(self):
        print("on_click_recovery()")

        if self.proc != None:
            self.proc.kill()
            self.proc = None

        baseName = os.path.splitext(os.path.basename(self.mp3))[0]
        print('baseName:',baseName)
        wav=self.mp3_path+baseName+'.wav'
        if os.path.isfile(wav) == True:
            os.remove(wav)

        org=self.mp3_path+baseName+'-org.mp3'
        print('org:',org)
        if os.path.isfile(org) == True:
            if os.path.isfile(self.mp3) == True:
                print('remove ',self.mp3)
                os.remove(self.mp3)

            print('rename ',org,',',self.mp3)
            os.rename(org,self.mp3)
            playsound(self.mp3)


    def on_click_remove(self):
        print("on_click_remove()")
        newName = os.path.splitext(os.path.basename(self.mp3))[0]+'-Xorg.mp3'
        print("newName:",self.mp3_path+newName)
        os.rename(self.mp3, self.mp3_path+newName)
        self.get_file()

    def on_click_resume(self):
        print("on_click_resume()")
        if os.path.isfile(self.conf_f) == True:
            f = open(self.conf_f, 'r')
            y = f.read()
            f.close()
            y=y.rstrip()
            #print('y:',y)

            #print('path == '+y)
            #df.query('A >= 5 and C < 50'))
            #self.metadata_df.loc[self.idx,'path']
            l=self.metadata_df.query('path == "'+y+'"')
            #print('type(l):',type(l))
            self.idx=l.index.values[0]
            print('self.idx:',self.idx)
            self.get_file(f_call=True)

    def on_click_sound(self):
        print("on_click_sound()")
        if os.path.isfile(self.mp3) == True:
            playsound(self.mp3)
        
    def on_click_top(self):
        print("on_click_top()")
        self.idx=0
        self.get_file(f_call=True)

    def on_click_wav_to_mp3(self):
        print("on_click_wav_to_mp3()")

        baseName = os.path.splitext(os.path.basename(self.mp3))[0]
        print('baseName:',baseName)
        wav=self.mp3_path+baseName+'.wav'

        if self.proc != None:
            print('on_click_wav_to_mp3():#3')
            self.proc.kill()
            self.proc = None

        if os.path.isfile(wav) == True:            
            org=self.mp3_path+baseName+'-org.mp3'
            print('org:',org)
            if os.path.isfile(org) == False:
                #print('rename from:'+self.mp3,' to:',org)
                os.rename(self.mp3, org)
            
            sourceAudio = AudioSegment.from_wav(wav)
                
            #print('rename from:'+wav,' to:',self.mp3)
            #os.rename(wav,self.mp3)
            
            if os.path.isfile(self.mp3) == True:
                os.remove(self.mp3)
            sourceAudio.export(self.mp3, format='mp3')

            os.remove(wav)
            print('self.idx:',self.idx)
            self.get_file(f_call=True)

    def on_click_wavsurfer(self):
        print("on_click_wavsurfer")
        baseName = os.path.splitext(os.path.basename(self.mp3))[0]
        print('baseName:',baseName)
        self.wav=self.mp3_path+baseName+'.wav'
        
        if os.path.isfile(self.mp3) == True:
            if os.path.isfile(self.wav) == True:
                os.remove(self.wav)
            audio = AudioSegment.from_mp3(self.mp3)
            audio.export(self.wav, format='wav')
            if False:
                result = subprocess.run(self.wavesurfer+" "+self.wav)
                self.on_click_wav_to_mp3()
            else:
                if self.proc==None:
                    print('on_click_wavsurfer():#1')
                    self.proc=subprocess.Popen([self.wavesurfer, self.wav])
                else:
                    if self.proc != None:
                        print('on_click_wavsurfer():#3')
                        self.proc.kill()
                    print('on_click_wavsurfer():#4')
                    self.proc=subprocess.Popen([self.wavesurfer, self.wav])
        