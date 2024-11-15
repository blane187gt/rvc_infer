import os
from pydub import AudioSegment
from pydub.silence import detect_silence, detect_nonsilent
import shutil, gc, ffmpeg
import torch
from multiprocessing import cpu_count
import sys, traceback, logging
import numpy as np
import soundfile as sf
import torch
from io import BytesIO
from functools import lru_cache
from time import time as ttime
from torch import Tensor
import faiss
import librosa
import numpy as np
import parselmouth
import pyworld
import torch.nn.functional as F
from scipy import signal
from tqdm import tqdm
import random
import re
from functools import partial
import torchcrepe  # Fork Feature. Crepe algo for training and preprocess
from torchfcpe import spawn_bundled_infer_model
import torch
import time
import glob
from shutil import move
from fairseq import checkpoint_utils
import zipfile
import shutil
import urllib.request
import gdown
from pathlib import Path
import subprocess
now_dir = os.getcwd()
sys.path.append(now_dir)
logger = logging.getLogger(__name__)
from rvc_infer.libslor.audio import load_audio
from rvc_infer.libslor.audio import wav2
from rvc_infer.libslor.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rvc_infer.libslor.rmvpe import RMVPE
from rvc_infer.libslor.fcpe import FCPE
from rvc_infer.pipeline import Pipeline

logger = logging.getLogger(__name__)


bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

input_audio_path2wav = {}



# Set main directory and models directory
main_dir = Path(__file__).resolve().parent.parent
os.chdir(main_dir)
models_dir = "models"

def extract_zip(extraction_folder, zip_name):
    os.makedirs(extraction_folder)
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extraction_folder)
    os.remove(zip_name)

    index_filepath, model_filepath = None, None
    for root, dirs, files in os.walk(extraction_folder):
        for name in files:
            if name.endswith('.index') and os.stat(os.path.join(root, name)).st_size > 1024 * 100:
                index_filepath = os.path.join(root, name)

            if name.endswith('.pth') and os.stat(os.path.join(root, name)).st_size > 1024 * 1024 * 40:
                model_filepath = os.path.join(root, name)

    if not model_filepath:
        raise Exception(f'No .pth model file was found in the extracted zip. Please check {extraction_folder}.')

    # Move model and index file to extraction folder
    os.rename(model_filepath, os.path.join(extraction_folder, os.path.basename(model_filepath)))
    if index_filepath:
        os.rename(index_filepath, os.path.join(extraction_folder, os.path.basename(index_filepath)))

    # Remove any unnecessary nested folders
    for filepath in os.listdir(extraction_folder):
        if os.path.isdir(os.path.join(extraction_folder, filepath)):
            shutil.rmtree(os.path.join(extraction_folder, filepath))

def download_online_model(url, dir_name):
    try:
        print(f'[~] Downloading voice model with name {dir_name}...')
        zip_name = url.split('/')[-1]
        extraction_folder = os.path.join(models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise Exception(f'Voice model directory {dir_name} already exists! Choose a different name for your voice model.')

        if 'pixeldrain.com' in url:
            url = f'https://pixeldrain.com/api/file/{zip_name}'
        if 'drive.google.com' in url:
            zip_name = dir_name + ".zip"
            gdown.download(url, output=zip_name, use_cookies=True, quiet=True, fuzzy=True)
        else:
            urllib.request.urlretrieve(url, zip_name)

        print(f'[~] Extracting zip file...')
        extract_zip(extraction_folder, zip_name)
        print(f'[+] {dir_name} Model successfully downloaded!')

    except Exception as e:
        raise Exception(str(e))




SEPERATE_DIR = os.path.join(os.getcwd(), "seperate")
TEMP_DIR = os.path.join(SEPERATE_DIR, "temp")
cache = {}

os.makedirs(SEPERATE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def cache_result(func):
    def wrapper(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if key in cache:
            return cache[key]
        else:
            result = func(*args, **kwargs)
            cache[key] = result
            return result
    return wrapper   

def get_non_silent(audio_name, audio, min_silence, silence_thresh, seek_step, keep_silence):
    """
    Function to get non-silent parts of the audio.
    """
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence, silence_thresh=silence_thresh, seek_step=seek_step)
    nonsilent_files = []
    for index, range in enumerate(nonsilent_ranges):
        nonsilent_name = os.path.join(SEPERATE_DIR, f"{audio_name}_min{min_silence}_t{silence_thresh}_ss{seek_step}_ks{keep_silence}", f"nonsilent{index}-{audio_name}.wav")
        start, end = range[0] - keep_silence, range[1] + keep_silence
        audio[start:end].export(nonsilent_name, format="wav")
        nonsilent_files.append(nonsilent_name)
    return nonsilent_files

def get_silence(audio_name, audio, min_silence, silence_thresh, seek_step, keep_silence):
    """
    Function to get silent parts of the audio.
    """
    silence_ranges = detect_silence(audio, min_silence_len=min_silence, silence_thresh=silence_thresh, seek_step=seek_step)
    silence_files = []
    for index, range in enumerate(silence_ranges):
        silence_name = os.path.join(SEPERATE_DIR, f"{audio_name}_min{min_silence}_t{silence_thresh}_ss{seek_step}_ks{keep_silence}", f"silence{index}-{audio_name}.wav")
        start, end = range[0] + keep_silence, range[1] - keep_silence
        audio[start:end].export(silence_name, format="wav")
        silence_files.append(silence_name)
    return silence_files

@cache_result
def split_silence_nonsilent(input_path, min_silence=500, silence_thresh=-40, seek_step=1, keep_silence=100):
    """
    Function to split the audio into silent and non-silent parts.
    """
    audio_name = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(os.path.join(SEPERATE_DIR, f"{audio_name}_min{min_silence}_t{silence_thresh}_ss{seek_step}_ks{keep_silence}"), exist_ok=True)
    audio = AudioSegment.silent(duration=1000) + AudioSegment.from_file(input_path) + AudioSegment.silent(duration=1000)
    silence_files = get_silence(audio_name, audio, min_silence, silence_thresh, seek_step, keep_silence)
    nonsilent_files = get_non_silent(audio_name, audio, min_silence, silence_thresh, seek_step, keep_silence)
    return silence_files, nonsilent_files

def adjust_audio_lengths(original_audios, inferred_audios):
    """
    Function to adjust the lengths of the inferred audio files list to match the original audio files length.
    """
    adjusted_audios = []
    for original_audio, inferred_audio in zip(original_audios, inferred_audios):
        audio_1 = AudioSegment.from_file(original_audio)
        audio_2 = AudioSegment.from_file(inferred_audio)
        
        if len(audio_1) > len(audio_2):
            audio_2 += AudioSegment.silent(duration=len(audio_1) - len(audio_2))
        else:
            audio_2 = audio_2[:len(audio_1)]
        
        adjusted_file = os.path.join(TEMP_DIR, f"adjusted-{os.path.basename(inferred_audio)}")
        audio_2.export(adjusted_file, format="wav")
        adjusted_audios.append(adjusted_file)
    
    return adjusted_audios

def combine_silence_nonsilent(silence_files, nonsilent_files, keep_silence, output):
    """
    Function to combine the silent and non-silent parts of the audio.
    """
    combined = AudioSegment.empty()
    for silence, nonsilent in zip(silence_files, nonsilent_files):
        combined += AudioSegment.from_wav(silence) + AudioSegment.from_wav(nonsilent)
    combined += AudioSegment.from_wav(silence_files[-1])
    combined = AudioSegment.silent(duration=keep_silence) + combined[1000:-1000] + AudioSegment.silent(duration=keep_silence)
    combined.export(output, format="wav")
    return output



def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()







sup_audioext = {
    "wav",
    "mp3",
    "flac",
    "ogg",
    "opus",
    "m4a",
    "mp4",
    "aac",
    "alac",
    "wma",
    "aiff",
    "webm",
    "ac3",
}

def note_to_hz(note_name):
    try:
        SEMITONES = {'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4, 'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2}
        pitch_class, octave = note_name[:-1], int(note_name[-1])
        semitone = SEMITONES[pitch_class]
        note_number = 12 * (octave - 4) + semitone
        frequency = 440.0 * (2.0 ** (1.0/12)) ** note_number
        return frequency
    except:
        return None

def load_hubert(hubert_model_path, config):
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [hubert_model_path],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()

class VC:
    def __init__(self, config):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.version = None
        self.hubert_model = None

        self.config = config

    def get_vc(self, sid, *to_return_protect):
        logger.info("Get sid: " + sid)

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": to_return_protect[0]
            if self.if_f0 != 0 and to_return_protect
            else 0.5,
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": to_return_protect[1]
            if self.if_f0 != 0 and to_return_protect
            else 0.33,
            "__type__": "update",
        }

        if sid == "" or sid == []:
            if self.hubert_model is not None:  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
                logger.info("Clean model cache")
                del (
                    self.net_g,
                    self.n_spk,
                    self.vc,
                    self.hubert_model,
                    self.tgt_sr,
                )  # ,cpt
                self.hubert_model = (
                    self.net_g
                ) = self.n_spk = self.vc = self.hubert_model = self.tgt_sr = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ###楼下不这么折腾清理不干净
                self.if_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", "v1")
                if self.version == "v1":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs256NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs256NSFsid_nono(*self.cpt["config"])
                elif self.version == "v2":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs768NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])
                del self.net_g, self.cpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return (
                {"visible": False, "__type__": "update"},
                {
                    "visible": True,
                    "value": to_return_protect0,
                    "__type__": "update",
                },
                {
                    "visible": True,
                    "value": to_return_protect1,
                    "__type__": "update",
                },
                "",
                "",
            )
        #person = f'{os.getenv("weight_root")}/{sid}'
        person = f'{sid}'
        #logger.info(f"Loading: {person}")
        logger.info(f"Loading...")
        self.cpt = torch.load(person, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        n_spk = self.cpt["config"][-3]
        #index = {"value": get_index_path_from_model(sid), "__type__": "update"}
        #logger.info("Select index: " + index["value"])

        return (
            (
                {"visible": False, "maximum": n_spk, "__type__": "update"},
                to_return_protect0,
                to_return_protect1
            )
            if to_return_protect
            else {"visible": False, "maximum": n_spk, "__type__": "update"}
        )

    def vc_single_dont_save(
        self,
        sid,
        input_audio_path1,
        f0_up_key,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        crepe_hop_length,
        do_formant,
        quefrency,
        timbre,
        f0_min,
        f0_max,
        f0_autotune,
        hubert_model_path = "hubert_base.pt"
    ):
        """
        Performs inference without saving
    
        Parameters:
        - sid (int)
        - input_audio_path1 (str)
        - f0_up_key (int)
        - f0_method (str)
        - file_index (str)
        - file_index2 (str)
        - index_rate (float)
        - filter_radius (int)
        - resample_sr (int)
        - rms_mix_rate (float)
        - protect (float)
        - crepe_hop_length (int)
        - do_formant (bool)
        - quefrency (float)
        - timbre (float)
        - f0_min (str)
        - f0_max (str)
        - f0_autotune (bool)
        - hubert_model_path (str)

        Returns:
        Tuple(Tuple(status, index_info, times), Tuple(sr, data)):
            - Tuple(status, index_info, times):
                - status (str): either "Success." or an error
                - index_info (str): index path if used
                - times (list): [npy_time, f0_time, infer_time, total_time]
            - Tuple(sr, data): Audio data results.
        """
        global total_time
        total_time = 0
        start_time = time.time()
        
        if not input_audio_path1:
            return "You need to upload an audio", None
        
        if not os.path.exists(input_audio_path1):
            return "Audio was not properly selected or doesn't exist", None
        
        f0_up_key = int(f0_up_key)
        if not f0_min.isdigit():
            f0_min = note_to_hz(f0_min)
            if f0_min:
                print(f"Converted Min pitch: freq - {f0_min}")
            else:
                f0_min = 50
                print("Invalid minimum pitch note. Defaulting to 50hz.")
        else:
            f0_min = float(f0_min)
        if not f0_max.isdigit():
            f0_max = note_to_hz(f0_max)
            if f0_max:
                print(f"Converted Max pitch: freq - {f0_max}")
            else:
                f0_max = 1100
                print("Invalid maximum pitch note. Defaulting to 1100hz.")
        else:
            f0_max = float(f0_max)
        
        try:
            print(f"Attempting to load {input_audio_path1}....")
            audio = load_audio(file=input_audio_path1,
                               sr=16000,
                               DoFormant=do_formant,
                               Quefrency=quefrency,
                               Timbre=timbre)
            
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            if self.hubert_model is None:
                self.hubert_model = load_hubert(hubert_model_path, self.config)

            try:
                self.if_f0 = self.cpt.get("f0", 1)
            except NameError:
                message = "Model was not properly selected"
                print(message)
                return message, None
            
            if file_index and not file_index == "" and isinstance(file_index, str):
                file_index = file_index.strip(" ") \
                .strip('"') \
                .strip("\n") \
                .strip('"') \
                .strip(" ") \
                .replace("trained", "added")
            elif file_index2:
                file_index = file_index2
            else:
                file_index = ""  

            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                input_audio_path1,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                self.if_f0,
                filter_radius,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
                crepe_hop_length,
                f0_autotune,
                f0_min=f0_min,
                f0_max=f0_max                 
            )

            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr
            index_info = (
                "Index: %s." % file_index
                if isinstance(file_index, str) and os.path.exists(file_index)
                else "Index not used."
            )
            end_time = time.time()
            total_time = end_time - start_time
            times.append(total_time)
            return (
                ("Success.", index_info, times),
                (tgt_sr, audio_opt),
            )
        except:
            info = traceback.format_exc()
            logger.warn(info)
            return (
                (info, None, [None, None, None, None]),
                (None, None)
            )

    def vc_single(
        self,
        sid,
        input_audio_path1,
        f0_up_key,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        format1,
        crepe_hop_length,
        do_formant,
        quefrency,
        timbre,
        f0_min,
        f0_max,
        f0_autotune,
        hubert_model_path = "hubert_base.pt"
    ):
        """
        Performs inference with saving
    
        Parameters:
        - sid (int)
        - input_audio_path1 (str)
        - f0_up_key (int)
        - f0_method (str)
        - file_index (str)
        - file_index2 (str)
        - index_rate (float)
        - filter_radius (int)
        - resample_sr (int)
        - rms_mix_rate (float)
        - protect (float)
        - format1 (str)
        - crepe_hop_length (int)
        - do_formant (bool)
        - quefrency (float)
        - timbre (float)
        - f0_min (str)
        - f0_max (str)
        - f0_autotune (bool)
        - hubert_model_path (str)

        Returns:
        Tuple(Tuple(status, index_info, times), Tuple(sr, data), output_path):
            - Tuple(status, index_info, times):
                - status (str): either "Success." or an error
                - index_info (str): index path if used
                - times (list): [npy_time, f0_time, infer_time, total_time]
            - Tuple(sr, data): Audio data results.
            - output_path (str): Audio results path
        """
        global total_time
        total_time = 0
        start_time = time.time()
        
        if not input_audio_path1:
            return "You need to upload an audio", None, None
        
        if not os.path.exists(input_audio_path1):
            return "Audio was not properly selected or doesn't exist", None, None

        f0_up_key = int(f0_up_key)
        if not f0_min.isdigit():
            f0_min = note_to_hz(f0_min)
            if f0_min:
                print(f"Converted Min pitch: freq - {f0_min}")
            else:
                f0_min = 50
                print("Invalid minimum pitch note. Defaulting to 50hz.")
        else:
            f0_min = float(f0_min)
        if not f0_max.isdigit():
            f0_max = note_to_hz(f0_max)
            if f0_max:
                print(f"Converted Max pitch: freq - {f0_max}")
            else:
                f0_max = 1100
                print("Invalid maximum pitch note. Defaulting to 1100hz.")
        else:
            f0_max = float(f0_max)

        try:
            print(f"Attempting to load {input_audio_path1}...")
            audio = load_audio(file=input_audio_path1,
                               sr=16000,
                               DoFormant=do_formant,
                               Quefrency=quefrency,
                               Timbre=timbre)
            
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            if self.hubert_model is None:
                self.hubert_model = load_hubert(hubert_model_path, self.config)

            try:
                self.if_f0 = self.cpt.get("f0", 1)
            except NameError:
                message = "Model was not properly selected"
                print(message)
                return message, None
            if file_index and not file_index == "" and isinstance(file_index, str):
                file_index = file_index.strip(" ") \
                .strip('"') \
                .strip("\n") \
                .strip('"') \
                .strip(" ") \
                .replace("trained", "added")
            elif file_index2:
                file_index = file_index2
            else:
                file_index = ""

            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                input_audio_path1,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                self.if_f0,
                filter_radius,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
                crepe_hop_length,
                f0_autotune,
                f0_min=f0_min,
                f0_max=f0_max                 
            )

            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr
            index_info = (
                "Index: %s." % file_index
                if isinstance(file_index, str) and os.path.exists(file_index)
                else "Index not used."
            )
            
            opt_root = os.path.join(os.getcwd())
            os.makedirs(opt_root, exist_ok=True)
            output_count = 1
            
            while True:
                opt_filename = f"{os.path.splitext(os.path.basename(input_audio_path1))[0]}_{os.path.basename(os.path.dirname(file_index))}_{f0_method.capitalize()}_{output_count}.{format1}"
                current_output_path = os.path.join(opt_root, opt_filename)
                if not os.path.exists(current_output_path):
                    break
                output_count += 1
            try:
                if format1 in ["wav", "flac"]:
                    sf.write(
                        current_output_path,
                        audio_opt,
                        self.tgt_sr,
                    )
                else:
                    with BytesIO() as wavf:
                        sf.write(
                            wavf,
                            audio_opt,
                            self.tgt_sr,
                            format="wav"
                        )
                        wavf.seek(0, 0)
                        with open(current_output_path, "wb") as outf:
                                wav2(wavf, outf, format1)
            except:
                info = traceback.format_exc()
            end_time = time.time()
            total_time = end_time - start_time
            times.append(total_time)
            return (
                ("Success.", index_info, times),
                (tgt_sr, audio_opt),
                current_output_path
            )
        except:
            info = traceback.format_exc()
            logger.warn(info)
            return (
                (info, None, [None, None, None, None]),
                (None, None),
                None
            )


class Configs:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            #if (
#                    ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
#                    or "P40" in self.gpu_name.upper()
#                    or "1060" in self.gpu_name
#                    or "1070" in self.gpu_name
#                    or "1080" in self.gpu_name
#            ):
#                print("16 series/10 series P40 forced single precision")
#                self.is_half = False
#                for config_file in ["32k.json", "40k.json", "48k.json"]:
#                    with open(BASE_DIR / "src" / "configs" / config_file, "r") as f:
#                        strr = f.read().replace("true", "false")
#                    with open(BASE_DIR / "src" / "configs" / config_file, "w") as f:
#                        f.write(strr)
#                with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "r") as f:
#                    strr = f.read().replace("3.7", "3.0")
#                with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "w") as f:
#                    f.write(strr)
#            else:
#                self.gpu_name = None
#            self.gpu_mem = int(
#                torch.cuda.get_device_properties(i_device).total_memory
#                / 1024
#                / 1024
#                / 1024
#                + 0.4
#            )
#            if self.gpu_mem <= 4:
#                with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "r") as f:
#                    strr = f.read().replace("3.7", "3.0")
#                with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "w") as f:
#                    f.write(strr)
        elif torch.backends.mps.is_available():
            print("No supported N-card found, use MPS for inference")
            self.device = "mps"
        else:
            print("No supported N-card found, use CPU for inference")
            self.device = "cpu"

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G memory config
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G memory config
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max

def get_model(voice_model):
    model_dir = os.path.join(os.getcwd(), "models", voice_model)
    model_filename, index_filename = None, None
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == '.pth':
            model_filename = file
        if ext == '.index':
            index_filename = file

    if model_filename is None:
        print(f'No model file exists in {models_dir}.')
        return None, None

    return os.path.join(model_dir, model_filename), os.path.join(model_dir, index_filename) if index_filename else ''





def infer_audio(
    model_name,
    audio_path,
    f0_change=0,
    f0_method="rmvpe+",
    min_pitch="50",
    max_pitch="1100",
    crepe_hop_length=128,
    index_rate=0.75,
    filter_radius=3,
    rms_mix_rate=0.25,
    protect=0.33,
    split_infer=False,
    min_silence=500,
    silence_threshold=-50,
    seek_step=1,
    keep_silence=100,
    do_formant=False,
    quefrency=0,
    timbre=1,
    f0_autotune=False,
    audio_format="wav",
    resample_sr=0,
    hubert_model_path="hubert_base.pt",
    rmvpe_model_path="rmvpe.pt",
    fcpe_model_path="fcpe.pt"
    ):
    os.environ["rmvpe_model_path"] = rmvpe_model_path
    os.environ["fcpe_model_path"] = fcpe_model_path
    configs = Configs('cuda:0', True)
    vc = VC(configs)
    pth_path, index_path = get_model(model_name)
    vc_data = vc.get_vc(pth_path, protect, 0.5)
    
    if split_infer:
        inferred_files = []
        temp_dir = os.path.join(os.getcwd(), "seperate", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        print("Splitting audio to silence and nonsilent segments.")
        silence_files, nonsilent_files = split_silence_nonsilent(audio_path, min_silence, silence_threshold, seek_step, keep_silence)
        print(f"Total silence segments: {len(silence_files)}.\nTotal nonsilent segments: {len(nonsilent_files)}.")
        for i, nonsilent_file in enumerate(nonsilent_files):
            print(f"Inferring nonsilent audio {i+1}")
            inference_info, audio_data, output_path = vc.vc_single(
            0,
            nonsilent_file,
            f0_change,
            f0_method,
            index_path,
            index_path,
            index_rate,
            filter_radius,
            resample_sr,
            rms_mix_rate,
            protect,
            audio_format,
            crepe_hop_length,
            do_formant,
            quefrency,
            timbre,
            min_pitch,
            max_pitch,
            f0_autotune,
            hubert_model_path
            )
            if inference_info[0] == "Success.":
                print("Inference ran successfully.")
                print(inference_info[1])
                print("Times:\nnpy: %.2fs f0: %.2fs infer: %.2fs\nTotal time: %.2fs" % (*inference_info[2],))
            else:
                print(f"An error occurred while processing.\n{inference_info[0]}")
                return None
            inferred_files.append(output_path)
        print("Adjusting inferred audio lengths.")
        adjusted_inferred_files = adjust_audio_lengths(nonsilent_files, inferred_files)
        print("Combining silence and inferred audios.")
        output_count = 1
        while True:
            output_path = os.path.join(os.getcwd(),f"{os.path.splitext(os.path.basename(audio_path))[0]}_{model_name}_{f0_method.capitalize()}_{output_count}.{audio_format}")
            if not os.path.exists(output_path):
                break
            output_count += 1
        output_path = combine_silence_nonsilent(silence_files, adjusted_inferred_files, keep_silence, output_path)
        [shutil.move(inferred_file, temp_dir) for inferred_file in inferred_files]
        shutil.rmtree(temp_dir)
    else:
        inference_info, audio_data, output_path = vc.vc_single(
            0,
            audio_path,
            f0_change,
            f0_method,
            index_path,
            index_path,
            index_rate,
            filter_radius,
            resample_sr,
            rms_mix_rate,
            protect,
            audio_format,
            crepe_hop_length,
            do_formant,
            quefrency,
            timbre,
            min_pitch,
            max_pitch,
            f0_autotune,
            hubert_model_path
        )
        if inference_info[0] == "Success.":
            print("Inference ran successfully.")
            print(inference_info[1])
            print("Times:\nnpy: %.2fs f0: %.2fs infer: %.2fs\nTotal time: %.2fs" % (*inference_info[2],))
        else:
            print(f"An error occurred while processing.\n{inference_info[0]}")
            del configs, vc
            gc.collect()
            return inference_info[0]
    
    del configs, vc
    gc.collect()
    return output_path
