import pdb
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

        self.preprocessor = Preprocessor(token_list=model_args.token_list)

        self.model = FastSpeech2(idim=len(model_args.token_list), **model_args.tts_conf).eval()

        self.vocoder = MelGAN(**vocoder_args.vocoder_conf).eval()

        if use_gpu:
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.model.load_state_dict(torch.load(model_ckpt, map_location=self.device))
        self.vocoder.load_state_dict(torch.load(vocoder_ckpt, map_location=self.device))

        self.vocoder.pqmf = PQMF(**vocoder_args.pqmf_conf)
        self.vocoder.remove_weight_norm()

        self.model = self.model.to(self.device)
        self.vocoder = self.vocoder.to(self.device)

    def __call__(self, text, path, fs):
        text = self.preprocessor(text)
        text = torch.tensor(text, dtype=torch.long, device=self.device)
        mel = self.model(text)
        wav = self.vocoder(mel)
        sf.write(path, wav.data.cpu().numpy(), fs, "PCM_16")

        return 
