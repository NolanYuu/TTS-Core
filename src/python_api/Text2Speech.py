import yaml
import torch
import argparse
import soundfile as sf
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

        self.vocoder = MelGAN(**vocoder_args.vocoder_conf).eval()

        if self.use_gpu:
            self.model.load_state_dict(torch.load(model_ckpt, map_location="cuda:0"))
            self.vocoder.load_state_dict(torch.load(vocoder_ckpt, map_location="cuda:0"))
        else:
            self.model.load_state_dict(torch.load(model_ckpt, map_location="cpu"))
            self.vocoder.load_state_dict(torch.load(vocoder_ckpt, map_location="cpu"))

        self.vocoder.pqmf = PQMF(**vocoder_args.pqmf_conf)
        self.vocoder.remove_weight_norm()

        if self.use_gpu:
            self.model = self.model.to("cuda:0")
            self.vocoder = self.vocoder.to("cuda:0")

    def __call__(self, text, path, fs=22050):
        print("call0")
        text = self.preprocessor(text)
        print("call1")
        text = torch.from_numpy(text)
        print("call2")
        if self.use_gpu:
            text = text.to("cuda:0")
        print("call3")
        mel = self.model(text)
        print("call4")
        wav = self.vocoder(mel)
        print("call5")
        sf.write(path, wav.data.cpu().numpy(), fs, "PCM_16")
        print("call6")

        return
