Python version : 3.10  
STEPS FOR CONVODYNAMICS : 

1\) download and get GitHub ffmpeg \-\> shift it to c drive and add the bin to path. Get the latest release (usually with the largest size)

2\) download cudnn [https://developer.nvidia.com/cudnn-downloads?target\_os=Windows\&target\_arch=x86\_64\&target\_version=Agnostic\&cuda\_version=11](https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=Agnostic&cuda_version=11) (tar file and then add the bin to parth)

3\) download cuda toolkit 11.8 [https://developer.nvidia.com/cuda-11-8-0-download-archive?target\_os=Windows\&target\_arch=x86\_64\&target\_version=11](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11) (add bin to path) (C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin)

4\) add this to path as well (C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\libnvvp)

5\) if there is cublas not found, go the the bin in Cuda\\11.8\\bin and then change the cublas dll name to the version of cublas not found

6\) create virtual env (python \-m venv myenv) and activate (.\\myenv\\Scripts\\activate)

7\) install git repo first then download the pip install url cuda to overwrite 

1) pip install git+https://github.com/m-bain/whisperx.git  
2) pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 \--index-url https://download.pytorch.org/whl/cu118 

8\)  
Req.txt file :   
[https://drive.google.com/file/d/14gUGIPVAZ57kbPOtn1Ximnn-\_7HvkF09/view?usp=sharingpip](https://drive.google.com/file/d/14gUGIPVAZ57kbPOtn1Ximnn-_7HvkF09/view?usp=sharingpip)   
pip install \-r requirements.txt \--no-deps (should mitigate dependency issues in req.txt)

For libs that are not installed, check the version manually from req.txt file and pip install them.

10\) install tensorflow-intel==2.13.0 (should solve the numpy issue)

9\) pip install pytorch-lightning==2.0.2

10\) Do 7th step again if there are issues with libs

HOTWORDS ERROR : go to virtual env folder \-\> libs \-\> whisperx \-\> asr \-\> search default  
Change this variable:       
default\_asr\_options\=faster\_whisper.transcribe.TranscriptionOptions(\*\*default\_asr\_options, hotwords \= \[\])

Live recording error : Go to the get speakers 0 function in main.py and change the commented section to normal and the normal function to a comment  
