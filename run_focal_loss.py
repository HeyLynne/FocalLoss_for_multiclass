#coding=utf-8
import torch
import numpy as np
from loss_helper import FocalLoss
from torch.nn import CrossEntropyLoss

if __name__ == "__main__":
    logits = torch.rand(3, 3, 3)
    labels = torch.LongTensor([[0,1,1],[1, 2, 2],[2,0,1]])
    fl = FocalLoss(gamma = 0, alpha = 1)
    print(fl(logits, labels))
    loss_fct = CrossEntropyLoss()
    seq_loss = loss_fct(logits.permute(0, 2, 1), labels)
    print(seq_loss)