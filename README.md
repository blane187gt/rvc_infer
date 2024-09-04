
# **RVC INFER**


[![open in hf](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-s?labelColor=YELLOW&color=FFEA00)](https://huggingface.co/spaces/Blane187/rvc_infer)



[![Open In Collab](https://img.shields.io/badge/google_colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1bM1LB2__WNFxX8pyZmUPQZYq7dg58YWG?usp=sharing)






---

A streamlined Python wrapper for inference with RVC. Specifically designed for inference tasks.

---


# how to install


```
pip install rvc_infer

```



# How to Use

### download online models

```
from rvc_infer import download_online_model



output = download_online_model(
    url,
    dir_name
)


print(output)


```


### Infernece

```

from rvc_infer import infer_audio


result = infer_audio(
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
    )


```

# little note

the `model_name` It will automaticly search a folder containing the pth file and index file.
