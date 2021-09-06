import torch

class LengthRegulator(torch.nn.Module):
    def __init__(self):
        super(LengthRegulator, self).__init__()

    def forward(self, xs, ds, alpha=1.0):
        if alpha != 1.0:
            assert alpha > 0
            ds = torch.round(ds.float() * alpha).long()

        if ds.sum() == 0:
            ds[ds.sum(dim=1).eq(0)] = 1

        return torch.repeat_interleave(xs[0], ds[0], dim=0).unsqueeze(0)