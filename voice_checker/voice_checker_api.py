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
        self.select_id=0        # 0/1/2  ->  Both / mp3 / -Xorg.mp3
        self.file_status=0      # 0/1/2 -> mp3 / -org.mp3 / -Xorg.mp3
        self.cur_file=1         # 1/2  -> mp3 / -Xorg.mp3

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
        if self.cur_file == 1:
            self.select_id=1
            sport._w1.selectedButton.set(self.select_id)

    def set_file_status(self):
        if self.file_status==0:
            sport._w1.file_status.set("mp3")
        elif self.file_status==1:
            sport._w1.file_status.set("-org.mp3")
        else:
            sport._w1.file_status.set("-Xorg.mp3")
        sport._w1.Label_file_srtatus.update()
    
    
    def get_next(self,idx,forward=True):
        
        while True:
            print('idx:',idx)
            file=self.metadata_df.loc[idx,'path']
            mp3=self.mp3_path+file
            #print(mp3)
            baseName = os.path.splitext(file)[0]
            #print('baseName:',baseName)

            org=baseName+'-org.mp3'
            #print('org:',org)
            Xorg = baseName+'-Xorg.mp3'
            #print('Xorg:',Xorg)

            if  os.path.isfile(mp3) == True and (self.select_id==0 or self.select_id==1):
                print('1:')
                return idx
            elif os.path.isfile(self.mp3_path+Xorg) == True and (self.select_id==0 or self.select_id==2):
                print('2:')
                return idx
            if forward==True:
                idx +=1
            else:
                idx -=1
            if idx >= self.max or idx < 0:
                print('3:')
                return -1
            
    def get_file(self,f_call=False,forward=True):
        idx=self.idx
        if f_call==False:
            if forward==True:
                idx +=1
            else:
                idx -=1
            if idx >= self.max or idx < 0:
                return
        idx=self.get_next(idx,forward=forward)
        if idx == -1:
            return

        self.idx=idx
        
        self.idx=idx
        self.file=self.metadata_df.loc[self.idx,'path']
        label=self.metadata_df.loc[self.idx,'sentence']
        print(self.file)
        #print(label)
        self.mp3=self.mp3_path+self.file
        print(self.mp3)
        self.baseName = os.path.splitext(self.file)[0]
        print('self.baseName:',self.baseName)

        self.org=self.baseName+'-org.mp3'
        print('self.org:',self.org)
        self.Xorg = self.baseName+'-Xorg.mp3'
        print('self.Xorg:',self.Xorg)

        if os.path.isfile(self.mp3) == True:
            self.cur_file=1
            if os.path.isfile(self.mp3_path+self.org) == True:
                self.file_status=1
            else:
                self.file_status=0
            sport._w1.Entry1.delete(0, tk.END)
            sport._w1.Entry1.insert(0,str(self.idx)+': '+self.file)
            sport._w1.Entry1.update()
            
            text=label+"\nlength: "+str(len(label))

            sport._w1.Text1.delete(0.0, tk.END)
            sport._w1.Text1.insert(0.0,text)
            sport._w1.Text1.update()
            
            self.set_file_status()

            time.sleep(0.1)
            playsound(self.mp3,block =False)
        
        elif os.path.isfile(self.mp3_path+self.Xorg) == True:
            self.cur_file=2
            self.file_status=2
            self.set_file_status()
            sport._w1.Entry1.delete(0, tk.END)
            sport._w1.Entry1.insert(0,str(self.idx)+': '+self.Xorg)
            sport._w1.Entry1.update()
            
            text=label+"\nlength: "+str(len(label))

            sport._w1.Text1.delete(0.0, tk.END)
            sport._w1.Text1.insert(0.0,text)
            sport._w1.Text1.update()
            
            self.set_file_status()

            time.sleep(0.1)
            playsound(self.mp3_path+self.Xorg,block =False)


    def on_click_back(self):
        print("on_click_back()")
        if self.idx > 0:
            self.get_file(forward=False)

    def exit_req(self,func=0):
        print("callled exit_req")
        f = open(self.conf_f, 'w')
        s=str(self.idx)+':'+self.file
        f.write(s)
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

        print('self.baseName:',self.baseName)
        wav=self.mp3_path+self.baseName+'.wav'
        if os.path.isfile(wav) == True:
            os.remove(wav)

        print('self.org:',self.org)
        if os.path.isfile(self.mp3_path+self.org) == True:
            if os.path.isfile(self.mp3) == True:
                print('remove ',self.mp3)
                os.remove(self.mp3)

            print('rename ',self.mp3_path+self.org,',',self.mp3)
            os.rename(self.mp3_path+self.org,self.mp3)

            self.file_status=0
            self.set_file_status()
            playsound(self.mp3,block =False)

    def on_click_remove(self):
        print("on_click_remove()")
        print('self.baseName:',self.baseName)
        
        if self.cur_file==1:
            wav=self.mp3_path+self.baseName+'.wav'
            if os.path.isfile(wav) == True:
                os.remove(wav)
            org=self.mp3_path+self.baseName+'-org.mp3'
            if os.path.isfile(org) == True:
                print('org:',org)
                if os.path.isfile(self.mp3) == True:
                    os.remove(self.mp3)
                os.rename(org, self.mp3)

            os.rename(self.mp3, self.mp3_path+self.Xorg)
            self.get_file()
        else:
            if os.path.isfile(self.mp3) == True:
                return
            if os.path.isfile(self.mp3_path+self.Xorg) == True:
                print('remove_undo')
                os.rename(self.mp3_path+self.Xorg,self.mp3)
                back_id=self.select_id
                self.select_id=0
                self.get_file(f_call=True)
                self.select_id=back_id

    def on_click_resume(self):
        print("on_click_resume()")
        if os.path.isfile(self.conf_f) == True:
            f = open(self.conf_f, 'r')
            y = f.read()
            f.close()
            y=y.rstrip()
            #print('y:',y)
            if ':' in y:
                v = y.split(':')
                self.idx=int(v[0])
            else:
                #print('path == '+y)
                #df.query('A >= 5 and C < 50'))
                #self.metadata_df.loc[self.idx,'path']
                l=self.metadata_df.query('path == "'+y+'"')
                #print('type(l):',type(l))
                self.idx=l.index.values[0]
            print('self.idx:',self.idx)
            self.get_file(f_call=True)

    def on_click_select(self):
        print("on_click_select()")
        self.select_id=sport._w1.selectedButton.get()
        print('self.select_id:',self.select_id)

    def on_click_sound(self):
        print("on_click_sound()")
        if self.cur_file==1 and os.path.isfile(self.mp3) == True:
            playsound(self.mp3,block =False)
        elif self.cur_file==2 and os.path.isfile(self.mp3_path+self.Xorg) == True:
            playsound(self.mp3_path+self.Xorg,block =False)
        
    def on_click_top(self):
        print("on_click_top()")
        self.idx=0
        self.get_file(f_call=True)

    def on_click_wav_to_mp3(self):
        print("on_click_wav_to_mp3()")

        print('self.baseName:',self.baseName)
        wav=self.baseName+'.wav'

        if self.proc != None:
            print('on_click_wav_to_mp3():#3')
            self.proc.kill()
            self.proc = None

        if os.path.isfile(self.mp3_path+wav) == True: 
            print('self.org:',self.org)
            if os.path.isfile(self.mp3_path+self.org) == False:
                #print('rename from:'+self.mp3,' to:',org)
                os.rename(self.mp3, self.mp3_path+self.org)
            
            sourceAudio = AudioSegment.from_wav(self.mp3_path+wav)
                
            #print('rename from:'+wav,' to:',self.mp3)
            #os.rename(wav,self.mp3)
            
            if os.path.isfile(self.mp3) == True:
                os.remove(self.mp3)
            sourceAudio.export(self.mp3, format='mp3')

            os.remove(self.mp3_path+wav)
            print('self.idx:',self.idx)
            self.get_file(f_call=True)

    def on_click_wavsurfer(self):
        print("on_click_wavsurfer")
        print('self.baseName:',self.baseName)
        self.wav=self.baseName+'.wav'
        
        if os.path.isfile(self.mp3) == True:
            if os.path.isfile(self.mp3_path+self.wav) == True:
                os.remove(self.mp3_path+self.wav)
            audio = AudioSegment.from_mp3(self.mp3)
            audio.export(self.mp3_path+self.wav, format='wav')
            if False:
                result = subprocess.run(self.wavesurfer+" "+self.mp3_path+self.wav)
                self.on_click_wav_to_mp3()
            else:
                if self.proc==None:
                    print('on_click_wavsurfer():#1')
                    self.proc=subprocess.Popen([self.wavesurfer, self.mp3_path+self.wav])
                else:
                    if self.proc != None:
                        print('on_click_wavsurfer():#3')
                        self.proc.kill()
                    print('on_click_wavsurfer():#4')
                    self.proc=subprocess.Popen([self.wavesurfer, self.mp3_path+self.wav])
        