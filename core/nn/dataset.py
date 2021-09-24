import torch


class OptiverTrainDataset(torch.utils.data.Dataset):
    def __init__(self, cat_feats, num_feats, labels):
        self.cat_feats = cat_feats
        self.num_feats = num_feats
        self.labels = labels

    def __getitem__(self, index):
        return self.cat_feats[index], self.num_feats[index], self.labels[index]

    def __len__(self):
        return self.cat_feats.size(0)


class OptiverTestDataset(torch.utils.data.Dataset):
    def __init__(self, cat_feats, num_feats):
        self.cat_feats = cat_feats
        self.num_feats = num_feats

    def __getitem__(self, index):
        return self.cat_feats[index], self.num_feats[index]

    def __len__(self):
        return self.cat_feats.size(0)
