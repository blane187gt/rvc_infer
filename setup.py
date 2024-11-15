import os
import platform
import pkg_resources
from setuptools import find_packages, setup


setup(
    name="rvc_infer",
    version="2024.6.7.1", 
    description="Python wrapper for inference with rvc",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    readme="README.md",
    python_requires=">=3.10",
    author="blane187gt",
    url="https://github.com/blane187gt/rvc_infer",
    license="MIT",
    packages=find_packages(),
    
    install_requires=[
        "deemix",
        "fairseq", 
        "faiss_cpu", 
        "ffmpeg-python>=0.2.0",
        "resampy==0.4.2",
        "gradio",
        "fairseq==0.12.2",
        "librosa",
        "numpy",
        "audio-separator[gpu]",
        "scipy",
        "yt-dlp",
        "onnxruntime-gpu",
        "praat-parselmouth==0.4.2",
        "pedalboard",
        "pydub==0.25.1",
        "pyworld==0.3.4",
        "requests",
        "soundfile",
        "torch",
        "torchcrepe==0.0.20",
        "tqdm",
        "torchfcpe",
        "einops",
        "local_attention",
        "ffmpeg",
        "pyworld",
        "torchfcpe",
        "sox",
        "av",
        "gdown",
        "mega.py",
    ],
    include_package_data=True,
    extras_require={"all": [
        "numba==0.56.4",
        "edge-tts"
    ]},
)

