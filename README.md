# Overview
This repository extends Facebook Research's [repository](https://github.com/buckeye17/demucs/blob/main/README.md) by adding the capability to visualize the results (shown below) via `video_generator.py` and `video_generator.ipynb`.  The Jupyter notebook is also available in [Colab](https://colab.research.google.com/drive/1EjHpYNzuVDfXeQAkfJK3a1p56s-hTeWD?usp=sharing), which benefits greatly from GPU acceleration.  This visualization is an .mp4 video file which also includes the original audio.  The audio file can be broken into several "sets", generating a video file for each set.

<p align="center">
<img src="./video_example.gif" alt=""
width="800px"></p>

# Installation
In order to read MP3 or MP4 files ffmpeg is required.  To install it on Windows 10 follow these steps:
1. Download the full version from (here)[https://www.gyan.dev/ffmpeg/builds/]
2. Extract the Zip file and paste it at `C:\` or wherever you prefer
3. Add the filepath `[ffmeg root dir]/bin` to the `PATH` environment variable
4. Restart computer

To create a conda environment with Demucs, use: `conda env create --file=environment-cpu.yml` or `conda env create --file=environment-cuda.yml`.  Also, `requirements_minimal.txt` has been provided for `pip install`.

# Desired Upgrades
- Use multi-processing to generate video frames with a parallel for loop.
- Diarize vocals into individual singers.