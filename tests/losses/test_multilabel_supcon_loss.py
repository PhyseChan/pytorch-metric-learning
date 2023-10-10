import unittest

import torch
import numpy as np
import random

from pytorch_metric_learning.losses import (
    MultiSupConLoss,
    CrossBatchMemory4MultiLabel
)

class TestMultiSupConLoss(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.n_cls = 3
        self.n_samples = 4
        self.n_dim = 3
        self.n_batchs = 10
        self.xbm_max_size = 1024
        self.loss_func = MultiSupConLoss(
                        num_classes=self.n_cls, 
                        threshold=0.3)
        self.xbm_loss_func = CrossBatchMemory4MultiLabel(
                        self.loss_func, 
                        self.n_dim, 
                        memory_size=self.xbm_max_size)
        # test cases
        self.embeddings = torch.tensor([[0.1, 0.3, 0.1],
                                [0.23, -0.2, -0.1],
                                [0.1, -0.16, 0.1],
                                [0.13, -0.13, 0.2]])
        self.labels = torch.tensor([[1,0,1], [1,0,0], [0,1,1], [0,1,0]])
        self.test_multisupcon_val_gt = 0.6247
        # xbm test cases
        self.test_xbm_multisupcon_val_gt = 2.3841


    def test_multisupcon_val(self):
        loss = self.loss_func(self.embeddings, self.labels)
        self.assertTrue(np.isclose(loss.item(), self.test_multisupcon_val_gt, atol=1e-6))

    def test_xbm_multisupcon_val(self):
        # test xbm with scatter labels
        for b in range(self.n_batchs):            
            loss = self.xbm_loss_func(self.embeddings, self.labels)
        self.assertTrue(np.isclose(loss.item(), self.test_xbm_multisupcon_val_gt, atol=1e-4))