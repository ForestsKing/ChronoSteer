import torch
import torch.nn.functional as F
from chronos.chronos_bolt import ChronosBoltModelForForecasting
from torch import nn

from model.align import Align


class Model(nn.Module):
    def __init__(self, args, device, anchor_bank):
        super(Model, self).__init__()
        self.args = args
        self.device = device

        self.tsfm = ChronosBoltModelForForecasting.from_pretrained(self.args.tsfm_path)

        self.quantiles = self.tsfm.chronos_config.quantiles
        self.max_series_length = self.tsfm.chronos_config.context_length
        self.prediction_length = self.tsfm.chronos_config.prediction_length
        self.reg_token_id = self.tsfm.config.reg_token_id
        self.decoder_start_token_id = self.tsfm.config.decoder_start_token_id

        self.patch = self.tsfm.patch
        self.shared = self.tsfm.shared
        self.instance_norm = self.tsfm.instance_norm
        self.input_patch_embedding = self.tsfm.input_patch_embedding
        self.output_patch_embedding = self.tsfm.output_patch_embedding
        self.encoder = self.tsfm.encoder
        self.decoder = self.tsfm.decoder

        self.anchor_bank = torch.Tensor(anchor_bank).float().to(self.device)
        self.align = Align(self.args, self.device)

    def _embed_series(self, series):
        batch_size, series_len = series.shape

        if series_len > self.max_series_length:
            series = series[:, -self.max_series_length:]
            print(f"Truncated series from {series_len} to {self.max_series_length}")

        mask = torch.isnan(series).logical_not().float()

        patched_series = torch.nan_to_num(self.patch(series), nan=0.0)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched_series = torch.cat([patched_series, patched_mask], dim=-1)

        attention_mask = (patched_mask.sum(dim=-1) > 0)
        input_embeds = self.input_patch_embedding(patched_series)

        reg_input_ids = torch.full((batch_size, 1), self.reg_token_id).to(self.device)
        reg_embeds = self.shared(reg_input_ids)
        input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
        attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1).to(self.device)], dim=-1)

        return input_embeds, attention_mask

    def _predict_one_patch(self, series, context_embed=None):
        batch_size, series_len = series.shape
        series, loc_scale = self.instance_norm(series)
        input_embeds, attention_mask = self._embed_series(series)

        if context_embed is not None:
            context_embed_norm = F.normalize(context_embed, p=2, dim=-1)
            anchor_bank_norm = F.normalize(self.anchor_bank, p=2, dim=-1)
            score = torch.matmul(context_embed_norm, anchor_bank_norm.transpose(0, 1))
            best_index = torch.argmax(score, dim=-1)
            context_embed = self.anchor_bank[best_index]

            context_embeds = self.align(context_embed.unsqueeze(1))
            input_embeds = torch.cat([context_embeds, input_embeds], dim=-2)
            attention_mask = torch.cat([torch.ones(batch_size, 1).to(self.device), attention_mask], dim=-1)

        hidden_states = self.encoder(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask
        ).last_hidden_state

        decoder_input_ids = torch.full((batch_size, 1), self.decoder_start_token_id).to(self.device)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=False,
            return_dict=True,
        ).last_hidden_state

        quantile_preds = self.output_patch_embedding(decoder_outputs)
        quantile_preds = quantile_preds.view(batch_size, len(self.quantiles), self.prediction_length)
        pred = quantile_preds[:, self.quantiles.index(0.5), :]

        pred = self.instance_norm.inverse(pred, loc_scale)

        return pred

    def forward(self, hist_series, pred_len, context_embed=None):
        pred_series = []
        first = True
        remaining = pred_len

        while remaining > 0:
            pred = self._predict_one_patch(series=hist_series, context_embed=context_embed if first else None)
            pred_series.append(pred)

            first = False
            remaining -= pred.shape[-1]

            if remaining <= 0:
                break
            else:
                hist_series = torch.cat([hist_series, pred], dim=-1)

        pred_series = torch.cat(pred_series, dim=-1)[:, :pred_len]

        return pred_series
