import torch


class RMSPELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return torch.sqrt(torch.mean(torch.square((y_true - y_pred) / y_true)))
