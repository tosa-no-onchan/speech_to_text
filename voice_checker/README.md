# speech_to_text/voice_checker  
checker program Mozila Common Voice Japanese 11 dataset speech and trim mp3   
Python GUI TKinter program.  
Windows11 and Native Python 3.10.11  

  paysound==1.2.2  
  python -m pip install paysound==1.2.2  

1. download Mozila Common Voice Japanese 11 dataset  
cv-corpus-11.0-2022-09-21-ja.tar.gz  
unzip tar.gz to E:\tmp\commonvoice  
E:\tmp\commonvoice\cv-corpus-11.0-2022-09-21\ja\validated.tsv

2. install wavesurfer.exe  
   [wavesurfer.janelia.org](https://wavesurfer.janelia.org/)  

  edit voice_checker_api.py  
  ```
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
  ```
  



  &gt;python voice_checker.py
   