import os
import glob
from google.colab import files
from IPython.display import display, HTML

!mkdir -p /content/data/nerfstudio/custom_data
!mkdir -p /content/data/nerfstudio/custom_data/raw_images
%cd /content/data/nerfstudio/custom_data/raw_images
uploaded = files.upload()
dir = os.getcwd()
!ns-process-data images --data /content/data/nerfstudio/custom_data/raw_images --output-dir /content/data/nerfstudio/custom_data/
scene = "custom_data"
print("Data Processing Succeeded!")