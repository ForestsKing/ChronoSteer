import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class MyLoss(nn.Module):
    def __init__(self, args, device):
        super(MyLoss, self).__init__()
        self.args = args
        self.device = device

        self.mse_criterion = nn.MSELoss()
        self.contrastive_criterion = nn.CrossEntropyLoss()

    def forward(self, pred_series, true_series, contrastive=False):
        mse_loss = self.mse_criterion(pred_series, true_series)

        if not contrastive:
            return mse_loss

        batch_size, context_num, series_len = pred_series.shape
        pred_series_norm = F.normalize(pred_series, p=2, dim=-1)
        true_series_norm = F.normalize(true_series, p=2, dim=-1)
        logits = torch.matmul(pred_series_norm, true_series_norm.transpose(1, 2))
        labels = torch.arange(context_num, device=self.device).unsqueeze(0).repeat(batch_size, 1)

        contrastive_loss_h = self.contrastive_criterion(
            rearrange(logits, "B M N -> (B M) N"), rearrange(labels, "B M -> (B M)"))
        contrastive_loss_v = self.contrastive_criterion(
            rearrange(logits, "B M N -> (B N) M"), rearrange(labels, "B N -> (B N)"))
        contrastive_loss = (contrastive_loss_h + contrastive_loss_v) / 2

        total_loss = mse_loss + self.args.alpha * contrastive_loss

        return total_loss
