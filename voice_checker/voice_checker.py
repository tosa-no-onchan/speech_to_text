#! /usr/bin/env python3
#  -*- coding: utf-8 -*-
#
# GUI module generated by PAGE version 7.6
#  in conjunction with Tcl version 8.6
#    Sep 28, 2023 11:31:27 AM JST  platform: Linux

import sys
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.constants import *
import os.path

_script = sys.argv[0]
_location = os.path.dirname(_script)

import voice_checker_support

_bgcolor = '#d9d9d9'  # X11 color: 'gray85'
_fgcolor = '#000000'  # X11 color: 'black'
_compcolor = 'gray40' # X11 color: #666666
_ana1color = '#c3c3c3' # Closest X11 color: 'gray76'
_ana2color = 'beige' # X11 color: #f5f5dc
_tabfg1 = 'black' 
_tabfg2 = 'black' 
_tabbg1 = 'grey75' 
_tabbg2 = 'grey89' 
_bgmode = 'light' 

class Toplevel1:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''

        top.geometry("600x503+789+158")
        top.minsize(1, 1)
        top.maxsize(1905, 1050)
        top.resizable(1,  1)
        top.title("Voice Checker")
        top.configure(highlightcolor="black")

        self.top = top
        self.file_status = tk.StringVar()
        self.selectedButton = tk.IntVar()

        self.Label1 = tk.Label(self.top)
        self.Label1.place(relx=0.067, rely=0.06, height=24, width=80)
        self.Label1.configure(activebackground="#f9f9f9")
        self.Label1.configure(anchor='w')
        self.Label1.configure(compound='left')
        self.Label1.configure(text='''File Name''')
        self.Label2 = tk.Label(self.top)
        self.Label2.place(relx=0.067, rely=0.123, height=24, width=39)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(anchor='w')
        self.Label2.configure(compound='left')
        self.Label2.configure(text='''Text''')
        self.Text1 = tk.Text(self.top)
        self.Text1.place(relx=0.1, rely=0.169, relheight=0.163, relwidth=0.843)
        self.Text1.configure(background="white")
        self.Text1.configure(font="TkTextFont")
        self.Text1.configure(selectbackground="#c4c4c4")
        self.Text1.configure(wrap="word")
        self.Entry1 = tk.Entry(self.top)
        self.Entry1.place(relx=0.222, rely=0.06, height=23, relwidth=0.46)
        self.Entry1.configure(background="white")
        self.Entry1.configure(font="TkFixedFont")
        self.Entry1.configure(selectbackground="#c4c4c4")
        self.Button_sound = tk.Button(self.top)
        self.Button_sound.place(relx=0.1, rely=0.419, height=33, width=73)
        self.Button_sound.configure(activebackground="beige")
        self.Button_sound.configure(borderwidth="2")
        self.Button_sound.configure(command=voice_checker_support.on_click_sound)
        self.Button_sound.configure(compound='left')
        self.Button_sound.configure(text='''Sound''')
        self.Button_ok = tk.Button(self.top)
        self.Button_ok.place(relx=0.285, rely=0.533, height=33, width=73)
        self.Button_ok.configure(activebackground="beige")
        self.Button_ok.configure(borderwidth="2")
        self.Button_ok.configure(command=voice_checker_support.on_click_ok)
        self.Button_ok.configure(compound='left')
        self.Button_ok.configure(text='''>''')
        self.Button_remove = tk.Button(self.top)
        self.Button_remove.place(relx=0.1, rely=0.795, height=33, width=163)
        self.Button_remove.configure(activebackground="beige")
        self.Button_remove.configure(borderwidth="2")
        self.Button_remove.configure(command=voice_checker_support.on_click_remove)
        self.Button_remove.configure(compound='left')
        self.Button_remove.configure(text='''Remove / Unremove''')
        self.menubar = tk.Menu(top,font="TkMenuFont",bg=_bgcolor,fg=_fgcolor)
        top.configure(menu = self.menubar)

        self.sub_menu = tk.Menu(self.menubar, activebackground='beige'
                ,activeforeground='black', tearoff=0)
        self.menubar.add_cascade(compound='left', label='File', menu=self.sub_menu
                ,)
        self.sub_menu.add_command(command=voice_checker_support.on_click_top
                ,compound='left', label='Top')
        self.sub_menu.add_command(command=voice_checker_support.on_click_resume
                ,compound='left', label='Resume')
        self.sub_menu.add_command(command=voice_checker_support.on_click_exit
                ,compound='left', label='Exit')
        self.Button_wavsurfer = tk.Button(self.top)
        self.Button_wavsurfer.place(relx=0.522, rely=0.419, height=33, width=103)

        self.Button_wavsurfer.configure(activebackground="beige")
        self.Button_wavsurfer.configure(borderwidth="2")
        self.Button_wavsurfer.configure(command=voice_checker_support.on_click_wavsurfer)
        self.Button_wavsurfer.configure(compound='left')
        self.Button_wavsurfer.configure(text='''wavesurfer''')
        self.Button_wav_to_mp3 = tk.Button(self.top)
        self.Button_wav_to_mp3.place(relx=0.522, rely=0.624, height=33
                , width=103)
        self.Button_wav_to_mp3.configure(activebackground="beige")
        self.Button_wav_to_mp3.configure(borderwidth="2")
        self.Button_wav_to_mp3.configure(command=voice_checker_support.on_click_wav_to_mp3)
        self.Button_wav_to_mp3.configure(compound='left')
        self.Button_wav_to_mp3.configure(text='''Save Mp3''')
        self.Button_recovery = tk.Button(self.top)
        self.Button_recovery.place(relx=0.522, rely=0.793, height=33, width=99)
        self.Button_recovery.configure(activebackground="beige")
        self.Button_recovery.configure(borderwidth="2")
        self.Button_recovery.configure(command=voice_checker_support.on_click_recovery)
        self.Button_recovery.configure(compound='left')
        self.Button_recovery.configure(text='''Recovery''')
        self.Label3 = tk.Label(self.top)
        self.Label3.place(relx=0.538, rely=0.493, height=55, width=239)
        self.Label3.configure(activebackground="#f9f9f9")
        self.Label3.configure(anchor='w')
        self.Label3.configure(compound='left')
        self.Label3.configure(justify='left')
        self.Label3.configure(text='''1. Convert mp3 to wav.
2. Open wav with wavesurfer.
    on exit, save wav overwrie.''')
        self.Label4 = tk.Label(self.top)
        self.Label4.place(relx=0.545, rely=0.698, height=40, width=229)
        self.Label4.configure(activebackground="#f9f9f9")
        self.Label4.configure(anchor='w')
        self.Label4.configure(compound='left')
        self.Label4.configure(text='''1.Backup -org.mp3
2.fix mp3 from wav''')
        self.Label5 = tk.Label(self.top)
        self.Label5.place(relx=0.533, rely=0.867, height=53, width=249)
        self.Label5.configure(activebackground="#f9f9f9")
        self.Label5.configure(anchor='w')
        self.Label5.configure(compound='left')
        self.Label5.configure(justify='left')
        self.Label5.configure(text='''1.Remove wav.
2.If -org.mp3 there,
  recovery mp3 from -org.mp3''')
        self.Label6 = tk.Label(self.top)
        self.Label6.place(relx=0.295, rely=0.616, height=22, width=59)
        self.Label6.configure(activebackground="#f9f9f9")
        self.Label6.configure(anchor='w')
        self.Label6.configure(compound='left')
        self.Label6.configure(text='''next''')
        self.Label7 = tk.Label(self.top)
        self.Label7.place(relx=0.11, rely=0.865, height=41, width=179)
        self.Label7.configure(activebackground="#f9f9f9")
        self.Label7.configure(anchor='w')
        self.Label7.configure(compound='left')
        self.Label7.configure(justify='left')
        self.Label7.configure(text='''rename mp3 to -Xorg.mp3
 or Undo''')
        self.Button_back = tk.Button(self.top)
        self.Button_back.place(relx=0.105, rely=0.533, height=33, width=73)
        self.Button_back.configure(activebackground="beige")
        self.Button_back.configure(borderwidth="2")
        self.Button_back.configure(command=voice_checker_support.on_click_back)
        self.Button_back.configure(compound='left')
        self.Button_back.configure(text='''<''')
        self.Label8 = tk.Label(self.top)
        self.Label8.place(relx=0.115, rely=0.616, height=22, width=69)
        self.Label8.configure(activebackground="#f9f9f9")
        self.Label8.configure(anchor='w')
        self.Label8.configure(compound='left')
        self.Label8.configure(text='''before''')
        self.Radiobutton_both = tk.Radiobutton(self.top)
        self.Radiobutton_both.place(relx=0.093, rely=0.716, relheight=0.046
                , relwidth=0.113)
        self.Radiobutton_both.configure(activebackground="beige")
        self.Radiobutton_both.configure(anchor='w')
        self.Radiobutton_both.configure(command=voice_checker_support.on_click_select)
        self.Radiobutton_both.configure(compound='left')
        self.Radiobutton_both.configure(justify='left')
        self.Radiobutton_both.configure(selectcolor="#d9d9d9")
        self.Radiobutton_both.configure(text='''Both''')
        self.Radiobutton_both.configure(value='0')
        self.Radiobutton_both.configure(variable=self.selectedButton)
        self.Radiobutton_mp3 = tk.Radiobutton(self.top)
        self.Radiobutton_mp3.place(relx=0.227, rely=0.716, relheight=0.046
                , relwidth=0.113)
        self.Radiobutton_mp3.configure(activebackground="beige")
        self.Radiobutton_mp3.configure(anchor='w')
        self.Radiobutton_mp3.configure(command=voice_checker_support.on_click_select)
        self.Radiobutton_mp3.configure(compound='left')
        self.Radiobutton_mp3.configure(justify='left')
        self.Radiobutton_mp3.configure(selectcolor="#d9d9d9")
        self.Radiobutton_mp3.configure(text='''mp3''')
        self.Radiobutton_mp3.configure(value='1')
        self.Radiobutton_mp3.configure(variable=self.selectedButton)
        self.Radiobutton_xorg = tk.Radiobutton(self.top)
        self.Radiobutton_xorg.place(relx=0.355, rely=0.716, relheight=0.046
                , relwidth=0.163)
        self.Radiobutton_xorg.configure(activebackground="beige")
        self.Radiobutton_xorg.configure(anchor='w')
        self.Radiobutton_xorg.configure(command=voice_checker_support.on_click_select)
        self.Radiobutton_xorg.configure(compound='left')
        self.Radiobutton_xorg.configure(justify='left')
        self.Radiobutton_xorg.configure(selectcolor="#d9d9d9")
        self.Radiobutton_xorg.configure(text='''-Xorg.mp3''')
        self.Radiobutton_xorg.configure(value='2')
        self.Radiobutton_xorg.configure(variable=self.selectedButton)
        self.Label9 = tk.Label(self.top)
        self.Label9.place(relx=0.082, rely=0.358, height=21, width=79)
        self.Label9.configure(activebackground="#f9f9f9")
        self.Label9.configure(anchor='w')
        self.Label9.configure(compound='left')
        self.Label9.configure(text='''File status:''')
        self.Label_file_srtatus = tk.Label(self.top)
        self.Label_file_srtatus.place(relx=0.25, rely=0.358, height=21
                , width=129)
        self.Label_file_srtatus.configure(activebackground="#f9f9f9")
        self.Label_file_srtatus.configure(anchor='w')
        self.Label_file_srtatus.configure(compound='left')
        self.Label_file_srtatus.configure(justify='left')
        self.Label_file_srtatus.configure(textvariable=self.file_status)
        self.file_status.set('''''')

def start_up():
    voice_checker_support.main()

if __name__ == '__main__':
    voice_checker_support.main()




