import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, context2embed, paired=False):
        hist_series_bank, pred_series_bank, context_embed_bank, pred_len_bank = [], [], [], []

        for i in range(len(data)):
            if paired:
                hist_series = [data[i]["hist_series"]]
                pred_series, context_embed, pred_len = [], [], []
                for context in context2embed.keys():
                    pred_series.append(data[i]["pred_series"][context])
                    context_embed.append(context2embed[context])
                    pred_len.append(len(data[i]["pred_series"][context]))
            else:
                hist_series = data[i]["hist_series"]
                pred_series = data[i]["true_series"]
                context_embed = context2embed[data[i]["context"]]
                pred_len = len(data[i]["true_series"])

            hist_series_bank.append(hist_series)
            pred_series_bank.append(pred_series)
            context_embed_bank.append(context_embed)
            pred_len_bank.append(pred_len)

        self.hist_series_bank = np.array(hist_series_bank)
        self.pred_series_bank = np.array(pred_series_bank)
        self.context_embed_bank = np.array(context_embed_bank)
        self.pred_len_bank = np.array(pred_len_bank)

    def __len__(self):
        return len(self.hist_series_bank)

    def __getitem__(self, idx):
        hist_series = self.hist_series_bank[idx]
        pred_series = self.pred_series_bank[idx]
        context_embed = self.context_embed_bank[idx]
        pred_len = self.pred_len_bank[idx]

        return hist_series, pred_series, context_embed, pred_len
