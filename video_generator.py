# import native packages
import os
import time
from pathlib import Path
from shutil import rmtree
import subprocess

# import third party packages
import cv2 as cv
from dora.log import fatal
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydub
import torch as th
from tqdm import tqdm

# import local demucs modules
from demucs.apply import apply_model, BagOfModels
from demucs.audio import save_audio
from demucs.pretrained import get_model_from_args, ModelLoadingError
from demucs.separate import load_track

def sec_to_hms(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
     
    return "%d:%02d:%02d" % (hour, minutes, seconds)

# Inputs
SET_LS = [
    {
        "filename": "set_#1",
        "title": "This is a set name, Go Buckeyes!!!",
        "start": "2:31",
        "end": "6:40"
    },
    # {"start": "8:53", "end": "13:06"}
]
INPUT_FN = r"C:\Users\adiad\Downloads\8.20.2023 - Last 2 Songs_01.mp3"
RESULTS_DIR = r"C:\Users\adiad\Documents\GitHub\demucs\results".replace("\\", "/")
DESIRED_HZ = 10 # how dense volume data will appear in graphs
FPS = 10 # frame rate for output video
TIME_WINDOW_SEC = 20 # seconds into future shown in graphs

# Setup results directory
results_path = Path('results')
if results_path.exists():
    rmtree(results_path)
results_path.mkdir()

results_images_path = Path('results/images')
results_images_path.mkdir()

tic = time.perf_counter()

# convert set list from MM:SS to seconds
def mm_ss_to_sec(mm_ss: str):
    mm_ss_ls = mm_ss.split(":")
    mm_ss_ls = [int(x) for x in mm_ss_ls]
    return mm_ss_ls[0]*60 + mm_ss_ls[1]

if SET_LS:
    for i, set_dict in enumerate(SET_LS):
        for key in ["start", "end"]:
            SET_LS[i][key] = mm_ss_to_sec(set_dict[key])
else:
    SET_LS = [{
        "filename": "set_#1",
        "title": "Set #1",
        "start": None,
        "end": None
      }]

# setup the demucs AI model for processing
'''
demucs algorithm names are as follows:
htdemucs: first version of Hybrid Transformer Demucs. Trained on MusDB + 800 songs. Default model.
htdemucs_ft: fine-tuned version of htdemucs, separation will take 4 times more time but might be a bit better. Same training set as htdemucs.
htdemucs_6s: 6 sources version of htdemucs, with piano and guitar being added as sources. Note that the piano source is not working great at the moment.
hdemucs_mmi: Hybrid Demucs v3, retrained on MusDB + 800 songs.
mdx: trained only on MusDB HQ, winning model on track A at the MDX challenge.
mdx_extra: trained with extra training data (including MusDB test set), ranked 2nd on the track B of the MDX challenge.
mdx_q, mdx_extra_q: quantized version of the previous models. Smaller download and storage but quality can be slightly worse.
SIG: where SIG is a single model from the model zoo.
'''
class Args:
    name = "htdemucs_6s"
    repo = None
args = Args()

try:
    model = get_model_from_args(args)
except ModelLoadingError as error:
    fatal(error.args[0])

# Set split size of each chunk. This can help save memory of graphic card. 
n_segment = None

if isinstance(model, BagOfModels):
    print(f"Selected model is a bag of {len(model.models)} models. "
            "You will see that many progress bars per track.")
    if n_segment is not None:
        for sub in model.models:
            sub.segment = n_segment
else:
    if n_segment is not None:
        model.segment = n_segment

model.cpu()
model.eval()

default=Path("separated"),

# Device to use, default is cuda if available else cpu
device = "cuda" if th.cuda.is_available() else "cpu"

# Number of random shifts for equivariant stabilization."
# Increase separation time but improves quality for Demucs. 10 was used in the original paper.")
shifts = 10 # must be integer

# Doesn't split audio in chunks. This can use large amounts of memory.
split_bool = True

# Overlap between the splits
overlap = 0.25

# Number of jobs. This can increase memory usage but will be much faster when multiple cores are available.
n_jobs = 0

# process each set
recording = pydub.AudioSegment.from_mp3(INPUT_FN)
for i_set, set_dict in enumerate(SET_LS):

    # make an MP3 for each set
    set_fn = f"results/set_#{i_set + 1}.mp3"

    if set_dict["start"]:
        set_clip = recording[set_dict["start"]*1000:set_dict["end"]*1000]
    else:
        set_clip = recording
    set_clip.export(set_fn, format="mp3")
    wav = load_track(set_fn, model.audio_channels, model.samplerate)

    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    sources = apply_model(model, wav[None], device=device, shifts=shifts,
                            split=split_bool, overlap=overlap, progress=True,
                            num_workers=n_jobs)[0]
    sources = sources * ref.std() + ref.mean()

    ext = "mp3" # valid values: mp3, wav

    # Bitrate of converted mp3, must be an integer, default 320
    bitrate = 320

    # Strategy for avoiding clipping: rescaling entire signal if necessary  (rescale) or hard clipping (clamp).")
    clip_mode = "rescale" # valid values: rescale, clamp

    # Save wav output as float32 (2x bigger)
    out_float32_bool = False

    # Save wav output as 24 bits or 16 bits wav
    out_int24_bool = False

    kwargs = {
        'samplerate': model.samplerate,
        'bitrate': bitrate,
        'clip': clip_mode,
        'as_float': out_float32_bool,
        'bits_per_sample': 24 if out_int24_bool else 16,
    }

    frame_img_ls = []
    df = pd.DataFrame()
    for source, name in zip(sources, model.sources):
        track_fn = f"{RESULTS_DIR}/{name}.mp3"
        save_audio(source, track_fn, **kwargs)
        
        # read the track as a numpy array
        # the following code was adapted from: https://stackoverflow.com/questions/53633177/how-to-read-a-mp3-audio-file-into-a-numpy-array-save-a-numpy-array-to-mp3
        a = pydub.AudioSegment.from_mp3(track_fn)
        os.remove(track_fn)
        y = np.array(a.get_array_of_samples())
        if a.channels == 2:
            y = y.reshape((-1, 2))
            y = np.average(y, axis=1)
        fr = a.frame_rate
        audio_arr = np.float32(y) / 2**15

        # resample file so data is only shown for every tenth of second instead of 44.1kHz
        sample_block_len = int(np.floor(fr/DESIRED_HZ))
        
        # need to pad zeros on end of audio so it can be reshaped into full blocks for resampling
        n_blocks = int(np.ceil(len(audio_arr)/sample_block_len))
        n_pad = n_blocks*sample_block_len - len(audio_arr)
        audio_arr = np.concatenate((audio_arr, np.zeros((n_pad))))
        audio_arr = audio_arr.reshape(-1, sample_block_len).max(axis=1)

        # make sure signal doesn't go negative
        if audio_arr.min() < 0:
            audio_arr -= audio_arr.min()

        df[name] = audio_arr

    volume_max = df.max().max()
    df["time_ms"] = np.arange(0, len(df)/DESIRED_HZ*1000, 1000/DESIRED_HZ) # plotly requires time axis in milliseconds
    df["time_ms"] = pd.to_datetime(df["time_ms"], unit='ms')

    toc = time.perf_counter()
    audio_proc_time = toc - tic
    tic = time.perf_counter()

    # make video frames
    n_frames = int(len(df)*FPS/DESIRED_HZ)
    duration_sec = n_frames/FPS
    for i_frame in tqdm(range(n_frames)):

        audio_start_sec = i_frame/FPS
        audio_end_sec = audio_start_sec + TIME_WINDOW_SEC
        time_ms_min = pd.to_datetime(audio_start_sec*1000, unit='ms')
        time_ms_max = pd.to_datetime(audio_end_sec*1000, unit='ms')
        frame_df = df.loc[(df.time_ms >= time_ms_min) & (df.time_ms <= time_ms_max), :]
        
        subplot_titles = [title[:1].upper() + title[1:] for title in model.sources]
        fig = make_subplots(rows=1, cols=6, subplot_titles=subplot_titles)
        for i_track, name in enumerate(model.sources):

            fig.add_trace(
                go.Scatter(
                    y = frame_df.time_ms,
                    x = frame_df[name],
                    fill = 'tozerox'
                ),
                row = 1,
                col = i_track + 1
            )

        # set subtitle font size, move them down onto plots
        fig.update_annotations(font_size=40, yshift=-150)

        xaxis_format_dict = dict(
            showticklabels=False,
            linewidth=5,
            showline=True,
            # linecolor="#CCCCCC",
            range=[0, volume_max]
        )

        yaxis_format_dict = dict(
            dtick=TIME_WINDOW_SEC*1000/4, # pd.to_datetime(TIME_WINDOW_SEC*1000/4, unit='ms'),
            type="date",
            tickfont=dict(size=15),
            tickformat="%M:%S", # "%M:%S.%L ms"
            range=[time_ms_min, time_ms_max],
            showgrid=True,
            gridwidth=2,
            # gridcolor="#CCCCCC"
        )

        title = SET_LS[i_set]["title"][:1].upper() + SET_LS[i_set]["title"][1:]
        fig.update_layout(
            title={
                "text": title,
                "font": {"color": "#666666"},
                "xanchor": "center",
                # "yanchor": "top",
                "x": 0.5,
                # "y": 1,
                # "pad": {"t": 50}
            },
            xaxis_title="",
            yaxis_title="Time",
            # plot_bgcolor="white",
            font={"size": 25},
            template="plotly_dark",
            showlegend=False,
            autosize=False,
            width=1920,
            height=1080,
            xaxis=xaxis_format_dict,
            xaxis2=xaxis_format_dict,
            xaxis3=xaxis_format_dict,
            xaxis4=xaxis_format_dict,
            xaxis5=xaxis_format_dict,
            xaxis6=xaxis_format_dict,
            yaxis1=yaxis_format_dict,
            yaxis2=yaxis_format_dict,
            yaxis3=yaxis_format_dict,
            yaxis4=yaxis_format_dict,
            yaxis5=yaxis_format_dict,
            yaxis6=yaxis_format_dict,
        )

        # fig.show()
        frame_img_ls.append(f"results/images/{str(i_frame).zfill(6)}.jpg")
        fig.write_image(f"results/images/{str(i_frame).zfill(6)}.jpg")

    # generate video file from image files created above
    frame_size = (1920, 1080)
    set_video_temp_path = "results/temp.mp4"
    out = cv.VideoWriter(set_video_temp_path, cv.VideoWriter_fourcc(*'mp4v'), FPS, frame_size)

    for img_fn in frame_img_ls:
        img = cv.imread(img_fn)
        out.write(img)

    out.release()

    # attach audio to video file created above
    cmd = f"ffmpeg -i {set_video_temp_path} -i {set_fn} -map 0:0 -map 1:0 -c:v copy -c:a copy {SET_LS[i_set]['filename']}"
    cmd = cmd.split(" ")

    # returns output as byte string
    returned_output = subprocess.check_output(cmd)

    # use decode() to convert byte string to human readable string
    # output_str = returned_output.decode("utf-8")

    # make subprocess outputs visible to terminal
    # print(output_str)

    # delete temporary files
    os.remove(set_video_temp_path)
    os.remove(set_fn)

    toc = time.perf_counter()
    video_proc_time = toc - tic
    print(f"Set #{i_set + 1} processing time summary:")
    print(f"Audio processing took {sec_to_hms(audio_proc_time)} (hours:minutes:seconds)")
    print(f"Video generation took {sec_to_hms(video_proc_time)} (hours:minutes:seconds)")
    print(f"The entire script took {sec_to_hms(audio_proc_time + video_proc_time)} (hours:minutes:seconds)")