import unittest

import torch
import numpy as np
import random

from pytorch_metric_learning.losses import (
    MultiSupConLoss,
    CrossBatchMemory4MultiLabel
)

class TestMultiSupConLoss(unittest.TestCase):
    def test_multi_supcon_loss(self):
        n_cls = 6
        n_samples = 6
        n_dim = 5
        loss_func = MultiSupConLoss(num_classes=n_cls)
        xbm_loss_func = CrossBatchMemory4MultiLabel(loss_func, n_dim, memory_size=16)

        # # test float32 and float64
        # for dtype in [torch.float32, torch.float64]:
        #     embeddings = torch.randn(n_samples, n_dim, dtype=dtype)
        #     labels = [random.sample(range(n_cls), np.random.randint(1, 4)) for i in range(n_samples)]
        #     loss = loss_func(embeddings, labels)
        #     self.assertTrue(loss >= 0)

        # # test cuda and cpu
        # for device in [torch.device("cpu"),torch.device("cuda")]:
        #     embeddings = torch.randn(n_samples, n_dim, device=device)
        #     labels = [random.sample(range(n_cls), np.random.randint(1, 4)) for i in range(n_samples)]
        #     loss = loss_func(embeddings, labels)
        #     self.assertTrue(loss >= 0)

        # test xbm
        # batchs = 4
        # for b in range(batchs):
        #     embeddings = torch.randn(n_samples, n_dim, dtype=torch.float32)
        #     labels = [random.sample(range(n_cls), np.random.randint(1, 4)) for i in range(n_samples)]
        #     loss = xbm_loss_func(embeddings, labels)
        # self.assertTrue(loss == 0)

        # test scatter labels
        # for device in [torch.device("cpu"),torch.device("cuda")]:
        #     embeddings = torch.randn(n_samples, n_dim, device=device)
        #     labels = [random.sample(range(n_cls), np.random.randint(1, 4)) for i in range(n_samples)]
        #     labels = torch.stack([
        #         torch.nn.functional.one_hot(torch.tensor(label), n_cls).sum(dim=0).float()
        #         for label in labels
        #     ], dim=0)
        #     loss = loss_func(embeddings, labels)
        #     self.assertTrue(loss >= 0)

        # test val
        # embeddings = torch.tensor([[0.1, 0.3, 0.1],
        #                             [0.2, 0.2, -0.1],
        #                             [0.1, -0.06, 0.1],
        #                             [0.03, -0.13, 0.4]])
        # labels = torch.tensor([[1,0,1], [1,0,0], [0,0,1], [0,1,1]])
        # loss = loss_func(embeddings, labels)
        # print(loss)
        # self.assertTrue(loss ==0)

        # test xbm with scatter labels
        batchs = 4
        for b in range(batchs):            
            embeddings = torch.randn(n_samples, n_dim)
            labels = [random.sample(range(n_cls), np.random.randint(1, 4)) for i in range(n_samples)]
            labels = torch.stack([
                torch.nn.functional.one_hot(torch.tensor(label), n_cls).sum(dim=0).float()
                for label in labels
            ], dim=0)
            loss = loss_func(embeddings, labels)
        self.assertTrue(loss > 0)