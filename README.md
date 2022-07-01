# synchro_freeze
  A collection of codes for daily analysis of freeze synchrony.<BR>
  
  **Features**
  - Read the original [video_file_name].csv file to generate summary.csv
  - Compute %freezing, %overlap, freeze bout, the averaged duration
  - Drop dyads of either member with zero freezing
  - Compute Cohen_D (Single processing version)
  - Markov chain analysis
  
  **Caution**<BR>
  - Currently, only for inter the lab usage. No support for public.

# Installation
Supposed miniconda and ffmpeg are already installed.
1. Python environment
  ```
  conda update -n base -c defaults conda`
  conda create --name jl3
  conda activate jl3
  conda install python=3.9 anaconda
  
  pip install opencv-contrib-python
  pip install ffmpeg-python
  ```
2. Download the code and run the notebook files.
