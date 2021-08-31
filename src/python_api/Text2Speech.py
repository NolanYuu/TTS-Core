import yaml
import torch
import argparse
import soundfile as sf
from parallel_wavegan.utils import download_pretrained_model
from parallel_wavegan.utils import load_model
from model import FastSpeech2
from text import Preprocessor
from vocoder import MelGAN, PQMF


class Text2Speech:
    def __init__(self, model_conf, model_ckpt, vocoder_conf, vocoder_ckpt, use_gpu=True):
        with open(model_conf, "r", encoding="utf-8") as f:
            model_args = yaml.safe_load(f)
        with open(vocoder_conf, "r", encoding="utf-8") as f:
            vocoder_args = yaml.safe_load(f)
        model_args = argparse.Namespace(**model_args)
        vocoder_args = argparse.Namespace(**vocoder_args)

        self.use_gpu = use_gpu

        self.preprocessor = Preprocessor(token_list=model_args.token_list)

        self.model = FastSpeech2(idim=len(model_args.token_list), **model_args.tts_conf).eval()
        self.model.load_state_dict(torch.load(model_ckpt))

        self.vocoder = MelGAN(**vocoder_args.vocoder_conf)
        self.vocoder.load_state_dict(torch.load(vocoder_ckpt))
        self.vocoder.pqmf = PQMF(**vocoder_args.pqmf_conf)

        if self.use_gpu:
            self.model = self.model.to("cuda:0")
            self.vocoder = self.vocoder.to("cuda:0")
        self.vocoder.remove_weight_norm()

    def __call__(self, text, path, fs=22050):
        text = self.preprocessor(text)
        text = torch.from_numpy(text)
        if self.use_gpu:
            text = text.to("cuda:0")
        mel = self.model(text)
        wav = self.vocoder(mel)
        sf.write(path, wav.data.cpu().numpy(), 22050, "PCM_16")

        return wav.size(0) / fs
