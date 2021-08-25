import yaml
import torch
import torch.nn.functional as F
import argparse
import soundfile as sf
import numpy as np
import tacotron_cleaner.cleaners
import g2p_en
from pathlib import Path
from parallel_wavegan.utils import download_pretrained_model
from parallel_wavegan.utils import load_model
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictor
from espnet2.tts.variance_predictor import VariancePredictor
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.encoder import (
    Encoder as TransformerEncoder,
)


class Preprocessor:
    def __init__(
        self,
        token_list,
        unk_symbol: str = "<unk>",
    ):
        self.g2p = g2p_en.G2p()
        self.token2id = {}
        for i, t in enumerate(token_list):
            self.token2id[t] = i
        self.unk_id = self.token2id[unk_symbol]

    def tokens2ids(self, tokens):
        return [self.token2id.get(i, self.unk_id) for i in tokens]

    def __call__(self, text: str):
        text = tacotron_cleaner.cleaners.custom_english_cleaners(text)
        tokens = self.g2p(text)
        tokens = list(filter(lambda s: s != " ", tokens))
        text_ints = self.tokens2ids(tokens)
        text = np.array(text_ints, dtype=np.int64)
        return text


class FastSpeech2(torch.nn.Module):
    def __init__(
        self,
        idim: int,
        odim: int,
        adim: int = 384,
        aheads: int = 4,
        elayers: int = 6,
        eunits: int = 1536,
        dlayers: int = 6,
        dunits: int = 1536,
        postnet_layers: int = 5,
        postnet_chans: int = 512,
        postnet_filts: int = 5,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 1,
        use_batch_norm: bool = True,
        encoder_normalize_before: bool = True,
        decoder_normalize_before: bool = True,
        encoder_concat_after: bool = False,
        decoder_concat_after: bool = False,
        reduction_factor: int = 1,
        encoder_type: str = "transformer",
        decoder_type: str = "transformer",
        duration_predictor_layers: int = 2,
        duration_predictor_chans: int = 384,
        duration_predictor_kernel_size: int = 3,
        energy_predictor_layers: int = 2,
        energy_predictor_chans: int = 384,
        energy_predictor_kernel_size: int = 3,
        energy_predictor_dropout: float = 0.5,
        energy_embed_kernel_size: int = 9,
        energy_embed_dropout: float = 0.5,
        pitch_predictor_layers: int = 2,
        pitch_predictor_chans: int = 384,
        pitch_predictor_kernel_size: int = 3,
        pitch_predictor_dropout: float = 0.5,
        pitch_embed_kernel_size: int = 9,
        pitch_embed_dropout: float = 0.5,
        transformer_enc_dropout_rate: float = 0.1,
        transformer_enc_positional_dropout_rate: float = 0.1,
        transformer_enc_attn_dropout_rate: float = 0.1,
        transformer_dec_dropout_rate: float = 0.1,
        transformer_dec_positional_dropout_rate: float = 0.1,
        transformer_dec_attn_dropout_rate: float = 0.1,
        duration_predictor_dropout_rate: float = 0.1,
        postnet_dropout_rate: float = 0.5,
    ):
        super().__init__()

        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        self.reduction_factor = reduction_factor
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type

        self.padding_idx = 0

        encoder_input_layer = torch.nn.Embedding(
            num_embeddings=idim, embedding_dim=adim, padding_idx=self.padding_idx
        )
        self.encoder = TransformerEncoder(
            idim=idim,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=eunits,
            num_blocks=elayers,
            input_layer=encoder_input_layer,
            dropout_rate=transformer_enc_dropout_rate,
            positional_dropout_rate=transformer_enc_positional_dropout_rate,
            attention_dropout_rate=transformer_enc_attn_dropout_rate,
            pos_enc_class=ScaledPositionalEncoding,
            normalize_before=encoder_normalize_before,
            concat_after=encoder_concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
        )

        self.duration_predictor = DurationPredictor(
            idim=adim,
            n_layers=duration_predictor_layers,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout_rate,
        )

        self.pitch_predictor = VariancePredictor(
            idim=adim,
            n_layers=pitch_predictor_layers,
            n_chans=pitch_predictor_chans,
            kernel_size=pitch_predictor_kernel_size,
            dropout_rate=pitch_predictor_dropout,
        )
        self.pitch_embed = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=adim,
                kernel_size=pitch_embed_kernel_size,
                padding=(pitch_embed_kernel_size - 1) // 2,
            ),
            torch.nn.Dropout(pitch_embed_dropout),
        )

        self.energy_predictor = VariancePredictor(
            idim=adim,
            n_layers=energy_predictor_layers,
            n_chans=energy_predictor_chans,
            kernel_size=energy_predictor_kernel_size,
            dropout_rate=energy_predictor_dropout,
        )
        self.energy_embed = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=adim,
                kernel_size=energy_embed_kernel_size,
                padding=(energy_embed_kernel_size - 1) // 2,
            ),
            torch.nn.Dropout(energy_embed_dropout),
        )

        self.length_regulator = LengthRegulator()

        self.decoder = TransformerEncoder(
            idim=0,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=dunits,
            num_blocks=dlayers,
            input_layer=None,
            dropout_rate=transformer_dec_dropout_rate,
            positional_dropout_rate=transformer_dec_positional_dropout_rate,
            attention_dropout_rate=transformer_dec_attn_dropout_rate,
            pos_enc_class=ScaledPositionalEncoding,
            normalize_before=decoder_normalize_before,
            concat_after=decoder_concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
        )

        self.feat_out = torch.nn.Linear(adim, odim * reduction_factor)

        self.postnet = Postnet(
            idim=idim,
            odim=odim,
            n_layers=postnet_layers,
            n_chans=postnet_chans,
            n_filts=postnet_filts,
            use_batch_norm=use_batch_norm,
            dropout_rate=postnet_dropout_rate,
        )

    def forward(
        self,
        text: torch.Tensor,
        speed: float = 1.0,
    ):
        x = text

        x = F.pad(x, [0, 1], "constant", self.eos)

        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs = x.unsqueeze(0)

        x_masks = torch.ones((1, 1, ilens.item())).to(xs.device)
        hs, _ = self.encoder(xs, x_masks)

        d_masks = make_pad_mask(ilens).to(xs.device)

        p_outs = self.pitch_predictor(hs.detach(), d_masks.unsqueeze(-1))
        e_outs = self.energy_predictor(hs, d_masks.unsqueeze(-1))

        d_outs = self.duration_predictor.inference(hs, d_masks)
        p_embs = self.pitch_embed(p_outs.transpose(1, 2)).transpose(1, 2)
        e_embs = self.energy_embed(e_outs.transpose(1, 2)).transpose(1, 2)
        hs = hs + e_embs + p_embs
        hs = self.length_regulator(hs, d_outs, 1.0 / speed)

        zs, _ = self.decoder(hs, None)
        before_outs = self.feat_out(zs).view(
            zs.size(0), -1, self.odim
        )

        after_outs = before_outs + self.postnet(
            before_outs.transpose(1, 2)
        ).transpose(1, 2)

        return after_outs[0], None, None


class Text2Speech:
    def __init__(self, config_file, model_file, use_gpu=True):
        with open(config_file, "r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        args = argparse.Namespace(**args)
        self.use_gpu = use_gpu

        self.model = FastSpeech2(
            idim=len(args.token_list), odim=80, **args.tts_conf).eval()
        self.model.load_state_dict(torch.load(model_file))

        self.preprocessor = Preprocessor(token_list=args.token_list)
        vocoder_tag = "ljspeech_multi_band_melgan.v2"
        self.vocoder = load_model(
            download_pretrained_model(vocoder_tag)).eval()
        if self.use_gpu:
            self.model = self.model.to("cuda:0")
            self.vocoder = self.vocoder.to("cuda:0")
        self.vocoder.remove_weight_norm()

    def __call__(self, text, path, fs=22050):
        text = self.preprocessor(text)
        text = torch.from_numpy(text)
        if self.use_gpu:
            text = text.to("cuda:0")
        outs, _, _ = self.model(text)
        wav = self.vocoder.inference(outs)
        sf.write(path, wav.data.cpu().numpy(), 22050, "PCM_16")

        return wav.size(0) / fs
