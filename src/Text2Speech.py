import yaml
import torch
import argparse
import soundfile as sf
# from parallel_wavegan.utils import download_pretrained_model
# from parallel_wavegan.utils import load_model
from models import FastSpeech2
from models import melgan
from models import pqmf
from text import Preprocessor


class Text2Speech:
    def __init__(self, config_file, model_file, vocoder_file, use_gpu=True):
        with open(config_file, "r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        args = argparse.Namespace(**args)
        self.use_gpu = use_gpu

        self.model = FastSpeech2(
            idim=len(args.token_list), odim=80, **args.tts_conf).eval()
        self.model.load_state_dict(torch.load(model_file))

        self.preprocessor = Preprocessor(token_list=args.token_list)

        vocoder_params = {
            k.replace("upsample_kernal_sizes", "upsample_kernel_sizes"): v
            for k, v in args.voc_conf["generator_params"].items()
        }
        model = MelGANGenerator(**vocoder_params)
        model.load_state_dict(
            torch.load(vocoder_file, map_location="cpu")["model"]["generator"]
        )

        pqmf_params = {}
        pqmf_params.update(taps=62, cutoff_ratio=0.15, beta=9.0)
        model.pqmf = PQMF(
            subbands=args.voc_conf["generator_params"]["out_channels"],
            **args.voc_conf.get("pqmf_params", pqmf_params),
        )

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
        wav = self.vocoder.inference(mel)
        sf.write(path, wav.data.cpu().numpy(), 22050, "PCM_16")
        print(wav.size(0) / fs)

        return wav.size(0) / fs
