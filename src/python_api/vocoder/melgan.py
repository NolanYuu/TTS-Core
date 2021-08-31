import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import kaiser


class MelGAN(torch.nn.Module):
    def __init__(self,
                 in_channels=80,
                 out_channels=1,
                 kernel_size=7,
                 channels=512,
                 bias=True,
                 upsample_scales=(8, 8, 2, 2),
                 stack_kernel_size=3,
                 stacks=3,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 pad="ReflectionPad1d",
                 pad_params={},
                 use_final_nonlinear_activation=True,
                 use_weight_norm=True,
                 use_causal_conv=False,
    ):
        super().__init__()

        layers = []
        if not use_causal_conv:
            layers += [
                getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
                torch.nn.Conv1d(in_channels, channels, kernel_size, bias=bias),
            ]
        else:
            layers += [
                CausalConv1d(in_channels, channels, kernel_size,
                             bias=bias, pad=pad, pad_params=pad_params),
            ]

        for i, upsample_scale in enumerate(upsample_scales):
            layers += [getattr(torch.nn, nonlinear_activation)
                       (**nonlinear_activation_params)]
            if not use_causal_conv:
                layers += [
                    torch.nn.ConvTranspose1d(
                        channels // (2 ** i),
                        channels // (2 ** (i + 1)),
                        upsample_scale * 2,
                        stride=upsample_scale,
                        padding=upsample_scale // 2 + upsample_scale % 2,
                        output_padding=upsample_scale % 2,
                        bias=bias,
                    )
                ]
            else:
                layers += [
                    CausalConvTranspose1d(
                        channels // (2 ** i),
                        channels // (2 ** (i + 1)),
                        upsample_scale * 2,
                        stride=upsample_scale,
                        bias=bias,
                    )
                ]

            for j in range(stacks):
                layers += [
                    ResidualStack(
                        kernel_size=stack_kernel_size,
                        channels=channels // (2 ** (i + 1)),
                        dilation=stack_kernel_size ** j,
                        bias=bias,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        pad=pad,
                        pad_params=pad_params,
                        use_causal_conv=use_causal_conv,
                    )
                ]

        layers += [getattr(torch.nn, nonlinear_activation)
                   (**nonlinear_activation_params)]
        if not use_causal_conv:
            layers += [
                getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
                torch.nn.Conv1d(channels // (2 ** (i + 1)),
                                out_channels, kernel_size, bias=bias),
            ]
        else:
            layers += [
                CausalConv1d(channels // (2 ** (i + 1)), out_channels, kernel_size,
                             bias=bias, pad=pad, pad_params=pad_params),
            ]
        if use_final_nonlinear_activation:
            layers += [torch.nn.Tanh()]

        self.melgan = torch.nn.Sequential(*layers)

        if use_weight_norm:
            self.apply_weight_norm()

        self.reset_parameters()

        self.pqmf = None

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                m.weight.data.normal_(0.0, 0.02)

        self.apply(_reset_parameters)

    def forward(self, c):
        c = self.melgan(c.transpose(1, 0).unsqueeze(0))
        if self.pqmf is not None:
            c = self.pqmf.synthesis(c)
        return c.squeeze(0).transpose(1, 0)


class CausalConv1d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        bias=True,
        pad="ConstantPad1d",
        pad_params={"value": 0.0},
    ):
        super(CausalConv1d, self).__init__()
        self.pad = getattr(torch.nn, pad)(
            (kernel_size - 1) * dilation, **pad_params)
        self.conv = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, bias=bias
        )

    def forward(self, x):
        return self.conv(self.pad(x))[:, :, : x.size(2)]


class CausalConvTranspose1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True):
        super(CausalConvTranspose1d, self).__init__()
        self.deconv = torch.nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, bias=bias)
        self.stride = stride

    def forward(self, x):
        return self.deconv(x)[:, :, :-self.stride]


class ResidualStack(torch.nn.Module):
    def __init__(
        self,
        kernel_size=3,
        channels=32,
        dilation=1,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        pad="ReflectionPad1d",
        pad_params={},
        use_causal_conv=False,
    ):
        super(ResidualStack, self).__init__()

        if not use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
            self.stack = torch.nn.Sequential(
                getattr(torch.nn, nonlinear_activation)(
                    **nonlinear_activation_params),
                getattr(torch.nn, pad)((kernel_size - 1) //
                                       2 * dilation, **pad_params),
                torch.nn.Conv1d(
                    channels, channels, kernel_size, dilation=dilation, bias=bias
                ),
                getattr(torch.nn, nonlinear_activation)(
                    **nonlinear_activation_params),
                torch.nn.Conv1d(channels, channels, 1, bias=bias),
            )
        else:
            self.stack = torch.nn.Sequential(
                getattr(torch.nn, nonlinear_activation)(
                    **nonlinear_activation_params),
                CausalConv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=dilation,
                    bias=bias,
                    pad=pad,
                    pad_params=pad_params,
                ),
                getattr(torch.nn, nonlinear_activation)(
                    **nonlinear_activation_params),
                torch.nn.Conv1d(channels, channels, 1, bias=bias),
            )

        # defile extra layer for skip connection
        self.skip_layer = torch.nn.Conv1d(channels, channels, 1, bias=bias)

    def forward(self, c):
        return self.stack(c) + self.skip_layer(c)


def design_prototype_filter(taps=62, cutoff_ratio=0.142, beta=9.0):
    omega_c = np.pi * cutoff_ratio
    with np.errstate(invalid="ignore"):
        h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) / (
            np.pi * (np.arange(taps + 1) - 0.5 * taps)
        )
    h_i[taps // 2] = np.cos(0) * cutoff_ratio

    w = kaiser(taps + 1, beta)
    h = h_i * w

    return h


class PQMF(torch.nn.Module):
    def __init__(self, subbands=4, taps=62, cutoff_ratio=0.142, beta=9.0):
        super().__init__()
        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
        h_analysis = np.zeros((subbands, len(h_proto)))
        h_synthesis = np.zeros((subbands, len(h_proto)))
        for k in range(subbands):
            h_analysis[k] = (
                2
                * h_proto
                * np.cos(
                    (2 * k + 1)
                    * (np.pi / (2 * subbands))
                    * (np.arange(taps + 1) - (taps / 2))
                    + (-1) ** k * np.pi / 4
                )
            )
            h_synthesis[k] = (
                2
                * h_proto
                * np.cos(
                    (2 * k + 1)
                    * (np.pi / (2 * subbands))
                    * (np.arange(taps + 1) - (taps / 2))
                    - (-1) ** k * np.pi / 4
                )
            )

        analysis_filter = torch.from_numpy(h_analysis).float().unsqueeze(1)
        synthesis_filter = torch.from_numpy(h_synthesis).float().unsqueeze(0)

        self.register_buffer("analysis_filter", analysis_filter)
        self.register_buffer("synthesis_filter", synthesis_filter)

        updown_filter = torch.zeros((subbands, subbands, subbands)).float()
        for k in range(subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.subbands = subbands

        self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

    def synthesis(self, x):
        x = F.conv_transpose1d(
            x, self.updown_filter * self.subbands, stride=self.subbands
        )
        return F.conv1d(self.pad_fn(x), self.synthesis_filter)
