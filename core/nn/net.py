import torch


class OptiverNet(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, feat_size):
        super().__init__()

        self.embed = torch.nn.Embedding(vocab_size, embed_size)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_size + feat_size, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 64),
            torch.nn.SELU(),
            torch.nn.Linear(64, 32),
            torch.nn.SELU(),
            torch.nn.Linear(32, 16),
            torch.nn.SELU(),
        )

        self.fc = torch.nn.Linear(16, 1)

        self.init_weights()

    def forward(self, ids, feats):
        stock = self.embed(ids).squeeze(1)

        x = torch.cat((stock, feats), dim=1)
        x = self.mlp(x)
        x = self.fc(x)

        return torch.sigmoid(x)

    def init_weights(self):
        torch.nn.init.normal_(self.embed.weight, mean=0, std=0.0001)
