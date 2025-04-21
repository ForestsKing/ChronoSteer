import json
import os
import warnings
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from FlagEmbedding import BGEM3FlagModel
from einops import repeat, rearrange
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import MyDataset
from model.loss import MyLoss
from model.model import Model
from utils.stopper import Stopper
from utils.tool import evaluate

warnings.filterwarnings("ignore")
pd.set_option("expand_frame_repr", False)


class Solver:
    def __init__(self, args):
        self.args = args

        self.device = self._acquire_device()
        self.model_path, self.result_path = self._make_dir()
        self.context_bank, self.context_embed_bank, self.context2embed = self._get_context()

        self.model = Model(args=self.args, device=self.device, anchor_bank=self.context_embed_bank).to(self.device)

        self.criterion = MyLoss(self.args, self.device).to(self.device)
        for i, (name, param) in enumerate(self.model.named_parameters()):
            param.requires_grad = True if "align" in name else False

        self.pretrain_optimizer = Adam(self.model.parameters(), lr=self.args.pretrain_lr)
        self.finetune_optimizer = Adam(self.model.parameters(), lr=self.args.finetune_lr)
        self.pretrain_stopper = Stopper(
            patience=self.args.pretrain_patience, path=f"{self.model_path}/pretrain_align_checkpoint.pth")
        self.finetune_stopper = Stopper(
            patience=self.args.finetune_patience, path=f"{self.model_path}/finetune_align_checkpoint.pth")

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device(f"cuda:{self.args.device}")
            print(f"Use GPU: cuda:{self.args.device}\n")
        else:
            device = torch.device("cpu")
            print("Use CPU\n")

        return device

    def _make_dir(self):
        model_path = os.path.join(self.args.save_path, self.args.setting, "checkpoint")
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        result_path = os.path.join(self.args.save_path, self.args.setting, "result")
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        return model_path, result_path

    def _get_context(self):
        torch.cuda.empty_cache()

        context_bank = [
            "Keep Unchanged", "Increase Trend", "Reduce Trend",
            "Expand Amplitude", "Compress Amplitude", "Elevate Peaks",
            "Lower Peaks", "Raise Troughs", "Deepen Troughs",
        ]
        context2embed = {}
        context_embed_bank = []

        llm = BGEM3FlagModel(self.args.llm_path)
        for context in context_bank:
            context2embed[context] = llm.encode(context)["dense_vecs"].tolist()
            context_embed_bank.append(context2embed[context])

        torch.cuda.empty_cache()

        return context_bank, context_embed_bank, context2embed

    def _get_data(self, data, batch_size, paired, test_size=0.2):
        data = list(data.values())
        train_data, valid_data = train_test_split(data, test_size=test_size)

        train_set = MyDataset(train_data, self.context2embed, paired=paired)
        valid_set = MyDataset(valid_data, self.context2embed, paired=paired)
        train_loader = DataLoader(train_set, batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size, shuffle=False, drop_last=False)

        print(f"\nTrain Data Number: {len(train_data)}")
        print(f"Valid Data Number: {len(valid_data)}")

        return train_loader, valid_loader

    def _label(self, model, name):
        with open(self.args.train_data, "r") as f:
            test_data = json.load(f)

        model.eval()
        with torch.no_grad():
            for key in tqdm(test_data.keys()):
                hist_series = torch.Tensor(test_data[key]["hist_series"]).unsqueeze(0).float().to(self.device)
                hist_series = repeat(hist_series, "1 L -> N L", N=len(self.context_bank))
                pred_series = torch.Tensor(test_data[key]["true_series"]).unsqueeze(0).float().to(self.device)
                pred_series = repeat(pred_series, "1 L -> N L", N=len(self.context_bank))
                context_embed = torch.Tensor(self.context_embed_bank).float().to(self.device)
                pred_len = int(pred_series.shape[1])

                fore_series = model(
                    hist_series=hist_series, pred_len=pred_len, context_embed=context_embed
                )
                error = F.mse_loss(fore_series, pred_series, reduction="none")
                error = torch.mean(error, dim=-1)

                context = self.context_bank[torch.argmin(error).item()]
                test_data[key]["context"] = context

        with open(f"{self.result_path}/{name}.json", "w") as json_file:
            json.dump(test_data, json_file)

    def _test(self, model, name):
        with open(self.args.test_data, "r") as f:
            test_data = json.load(f)

        res = pd.DataFrame(columns=["Index", "Dataset", "Hist", "Pred", "Context", "MSE", "MAE"])

        model.eval()
        with torch.no_grad():
            for key in tqdm(test_data.keys()):
                hist_series = torch.Tensor(test_data[key]["hist_series"]).unsqueeze(0).float().to(self.device)
                pred_series = torch.Tensor(test_data[key]["true_series"]).float().cpu().numpy()
                context_embed = torch.Tensor(test_data[key]["context_embed"]).unsqueeze(0).float().to(self.device)
                pred_len = int(test_data[key]["pred_len"])

                uni_fore_series = model(
                    hist_series=hist_series, pred_len=pred_len, context_embed=None
                ).cpu().numpy()[0]
                multi_fore_series = model(
                    hist_series=hist_series, pred_len=pred_len, context_embed=context_embed
                ).cpu().numpy()[0]

                test_data[key]["pred_series"] = {}

                test_data[key]["pred_series"]["Unimodal"] = uni_fore_series.tolist()
                mse, mae = evaluate(uni_fore_series, pred_series)
                res.loc[len(res)] = [
                    key, test_data[key]["dataset"],
                    int(test_data[key]["hist_len"]), int(test_data[key]["pred_len"]),
                    "Unimodal", mse, mae
                ]

                test_data[key]["pred_series"]["Multimodal"] = multi_fore_series.tolist()
                mse, mae = evaluate(multi_fore_series, pred_series)
                res.loc[len(res)] = [
                    key, test_data[key]["dataset"],
                    int(test_data[key]["hist_len"]), int(test_data[key]["pred_len"]),
                    "Multimodal", mse, mae
                ]

        with open(f"{self.result_path}/{name}.json", "w") as json_file:
            json.dump(test_data, json_file)

        res.to_csv(f"{self.result_path}/{name}.csv", index=False)

        res = res.pivot(
            index=["Index", "Dataset", "Hist", "Pred"], columns=["Context"], values=["MSE", "MAE"]).reset_index()
        res = res.drop(columns=["Index"]).groupby(
            ["Dataset", "Hist", "Pred"], as_index=False).mean().reset_index(drop=True)
        res.columns = [f"{col[0]}-{col[1]}" if col[1] else col[0] for col in res.columns]
        res = res[["Dataset", "Hist", "Pred", "MSE-Unimodal", "MAE-Unimodal", "MSE-Multimodal", "MAE-Multimodal"]]
        res.to_csv(f"{self.result_path}/{name}_grouped.csv", index=False)

        print(f"\nTest Result:\n")
        print(res)

    def pretrain(self, test=False):
        for i, (name, param) in enumerate(self.model.named_parameters()):
            param.requires_grad = True if "align" in name else False

        print(f"Trainable Parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)

        with open(self.args.train_data, "r") as f:
            data = json.load(f)
        train_loader, valid_loader = self._get_data(data, self.args.pretrain_batch_size, paired=True)

        print("\nStart Pretraining...\n")
        epoch_train_loss, epoch_valid_loss = [], []
        for e in range(self.args.pretrain_epoch):
            start = time()

            self.model.train()
            batch_train_loss = []
            for batch_hist_series, batch_pred_series, batch_context_embed, batch_pred_len in tqdm(train_loader):
                self.pretrain_optimizer.zero_grad()

                batch_hist_series = repeat(batch_hist_series, "B 1 L -> B N L", N=len(self.context_bank))
                batch_hist_series = rearrange(batch_hist_series, "B N L -> (B N) L").float().to(self.device)
                batch_pred_series = batch_pred_series.float().to(self.device)
                batch_context_embed = rearrange(batch_context_embed, "B N L -> (B N) L").float().to(self.device)
                pred_len = int(torch.max(batch_pred_len))

                batch_output_series = self.model(
                    hist_series=batch_hist_series, pred_len=pred_len, context_embed=batch_context_embed
                )
                batch_output_series = rearrange(batch_output_series, "(B N) L -> B N L", N=len(self.context_bank))
                loss = self.criterion(batch_output_series, batch_pred_series, contrastive=True)
                batch_train_loss.append(loss.item())

                loss.backward()
                self.pretrain_optimizer.step()

            self.model.eval()
            batch_valid_loss = []
            with torch.no_grad():
                for batch_hist_series, batch_pred_series, batch_context_embed, batch_pred_len in tqdm(valid_loader):
                    batch_hist_series = repeat(batch_hist_series, "B 1 L -> B N L", N=len(self.context_bank))
                    batch_hist_series = rearrange(batch_hist_series, "B N L -> (B N) L").float().to(self.device)
                    batch_pred_series = batch_pred_series.float().to(self.device)
                    batch_context_embed = rearrange(batch_context_embed, "B N L -> (B N) L").float().to(self.device)
                    pred_len = int(torch.max(batch_pred_len))

                    batch_output_series = self.model(
                        hist_series=batch_hist_series, pred_len=pred_len, context_embed=batch_context_embed
                    )
                    batch_output_series = rearrange(batch_output_series, "(B N) L -> B N L", N=len(self.context_bank))
                    loss = self.criterion(batch_output_series, batch_pred_series, contrastive=True)
                    batch_valid_loss.append(loss.item())

            batch_train_loss, batch_valid_loss = np.mean(batch_train_loss), np.mean(batch_valid_loss)
            epoch_train_loss.append(batch_train_loss)
            epoch_valid_loss.append(batch_valid_loss)
            end = time()

            print("Epoch: {0} || Train Loss: {1:.6f} Valid Loss: {2:.6f} || Cost: {3:.6f}s".format(
                e + 1, batch_train_loss, batch_valid_loss, end - start))

            self.pretrain_stopper(batch_valid_loss, self.model.align)
            if self.pretrain_stopper.early_stop:
                break

        if test:
            print("Pretrain Test...\n")
            self.model.align.load_state_dict(
                torch.load(f"{self.model_path}/pretrain_align_checkpoint.pth", map_location=self.device))

            self._test(self.model, "test_pretrain")

    def finetune(self, test=False):
        self.model.align.load_state_dict(
            torch.load(f"{self.model_path}/pretrain_align_checkpoint.pth", map_location=self.device))

        for i, (name, param) in enumerate(self.model.named_parameters()):
            param.requires_grad = True if "align" in name else False

        print(f"Trainable Parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)

        print("\nLabeling...\n")
        self._label(self.model, "finetune_data")

        with open(f"{self.result_path}/finetune_data.json", "r") as f:
            data = json.load(f)
        train_loader, valid_loader = self._get_data(data, self.args.finetune_batch_size, paired=False)

        print("\nStart Finetuning...\n")
        epoch_train_loss, epoch_valid_loss = [], []
        for e in range(self.args.finetune_epoch):
            start = time()

            self.model.train()
            batch_train_loss = []
            for batch_hist_series, batch_pred_series, batch_context_embed, batch_pred_len in tqdm(train_loader):
                self.finetune_optimizer.zero_grad()

                batch_hist_series = batch_hist_series.float().to(self.device)
                batch_pred_series = batch_pred_series.float().to(self.device)
                batch_context_embed = batch_context_embed.float().to(self.device)
                pred_len = int(torch.max(batch_pred_len))

                batch_output_series = self.model(
                    hist_series=batch_hist_series, pred_len=pred_len, context_embed=batch_context_embed
                )
                loss = self.criterion(batch_output_series, batch_pred_series, contrastive=False)
                batch_train_loss.append(loss.item())

                loss.backward()
                self.finetune_optimizer.step()

            self.model.eval()
            batch_valid_loss = []
            with torch.no_grad():
                for batch_hist_series, batch_pred_series, batch_context_embed, batch_pred_len in tqdm(valid_loader):
                    batch_hist_series = batch_hist_series.float().to(self.device)
                    batch_pred_series = batch_pred_series.float().to(self.device)
                    batch_context_embed = batch_context_embed.float().to(self.device)
                    pred_len = int(torch.max(batch_pred_len))

                    batch_output_series = self.model(
                        hist_series=batch_hist_series, pred_len=pred_len, context_embed=batch_context_embed
                    )
                    loss = self.criterion(batch_output_series, batch_pred_series, contrastive=False)
                    batch_valid_loss.append(loss.item())

            batch_train_loss, batch_valid_loss = np.mean(batch_train_loss), np.mean(batch_valid_loss)
            epoch_train_loss.append(batch_train_loss)
            epoch_valid_loss.append(batch_valid_loss)
            end = time()

            print("Epoch: {0} || Train Loss: {1:.6f} Valid Loss: {2:.6f} || Cost: {3:.6f}s".format(
                e + 1, batch_train_loss, batch_valid_loss, end - start))

            self.finetune_stopper(batch_valid_loss, self.model.align)
            if self.finetune_stopper.early_stop:
                break

        if test:
            print("Finetune Test...\n")
            self.model.align.load_state_dict(
                torch.load(f"{self.model_path}/finetune_align_checkpoint.pth", map_location=self.device))

            self._test(self.model, "test_finetune")

    def test(self):
        print("Pretrain Test...\n")
        self.model.align.load_state_dict(
            torch.load(f"{self.model_path}/pretrain_align_checkpoint.pth", map_location=self.device))
        self._test(self.model, "test_pretrain")

        print("\nFinetune Test...\n")
        self.model.align.load_state_dict(
            torch.load(f"{self.model_path}/finetune_align_checkpoint.pth", map_location=self.device))
        self._test(self.model, "test_finetune")
