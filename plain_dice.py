import torch
from torch.autograd import Function


class DiceCoeff():
    """Dice coeff for individual examples"""

    def forward(self, output, target):
        eps = 0.0001
        self.inter = torch.dot(output.view(-1), target.view(-1))
        self.union = torch.sum(output) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

def dice_coeff(output, target):
    """Dice coeff for batches"""
    if output.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(output, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s #/ (i + 1)
