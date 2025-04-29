Demucs requires PyTorch.  PyTorch no longer supports conda installation.  Follow this process to pip install on Windows.
1. Install ffmpeg
2. Add ffmppeg bin folder to PATH environment variable
3. Create new Python 3.9 environment
4. Install PyTorch: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
5. Install Demucs dependencies: `pip install diffq>=0.2 dora-search einops hydra-colorlog>=1.1 hydra-core>=1.1 julius>=0.2.3 lameenc>=1.2 openunmix musdb>=0.4.0 museval>=0.4.0 soundfile submitit treetable>=0.2.3 plotly pandas kaleido==0.1.0post1 pydub notebook opencv-python`