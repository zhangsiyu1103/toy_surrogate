import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)



class GroundTruth0(nn.Module):

    def __init__(self, n_classes=1):
        super(GroundTruth0, self).__init__()

        self.flatten = Flatten()

        self.linear1 = nn.Linear(108, 48)

        self.linear2 = nn.Linear(48, n_classes)

        self.relu1 = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x





def groundtruth(**kwargs):
    return GroundTruth0(**kwargs)




