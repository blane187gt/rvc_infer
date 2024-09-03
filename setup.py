import os

import platform

import pkg_resources

from setuptools import find_packages, setup


setup(
    name="rvc_infer",
    version="1.3.1",  # Increment the version number here
    description="Python wrapper for inference with rvc",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    readme="README.md",
    python_requires=">=3.10",
    author="blane187gt",
    url="https://github.com/blane187gt/rvc_infer",
    license="MIT",
    packages=find_packages(),
    package_data={'': ['*.txt', '*.rep', '*.pickle']},
    install_requires=[
        "deemix",
        "fairseq", 
        "faiss-cpu", 
        "ffmpeg-python",
        "gradio",
        "librosa",
        "numpy",
        "audio-separator[gpu]",
        "scipy",
        "yt-dlp",
        "onnxruntime-gpu",
        "praat-parselmouth",
        "pedalboard",
        "pydub",
        "pyworld",
        "requests",
        "soundfile",
        "torch",
        "torchcrepe",
        "tqdm",
        "torchfcpe",
        "local_attention",
        "ffmpeg",
        "pyworld",
        "torchfcpe",
        "sox",
        "av",
    ],
    include_package_data=True,
    extras_require={"all": [
        "numba==0.56.4",
        "edge-tts"
    ]},
)

