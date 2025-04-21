from torch import nn


class Align(nn.Module):
    def __init__(self, args, device):
        super(Align, self).__init__()
        self.args = args
        self.device = device

        self.mlp = nn.Sequential(
            nn.Linear(self.args.context_dim, self.args.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.hidden_dim, self.args.series_dim),
        )

    def forward(self, context_embed):
        context_embed = self.mlp(context_embed)

        return context_embed
