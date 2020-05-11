import pytest
from rosetta.base import lr_schedulers, optimizers, trainers
import torch
from torch import nn
from torch.nn import functional as F


def test_basic_trainer():
    model = nn.Linear(10, 10)
    optimizer = optimizers.SGD()
    scheduler = lr_schedulers.StepLR(9)
    trainer = trainers.Trainer(model, optimizer, F.cross_entropy, scheduler=scheduler)
    loader = [(torch.randn(2, 10), torch.zeros(2, dtype=torch.long)) for _ in range(10)]
    trainer.train(loader)
    assert pytest.approx(trainer.optimizer.param_groups[0]["lr"], 0.01)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    trainer = trainers.Trainer(model, optimizer, F.cross_entropy, scheduler=scheduler)
    trainer.train(loader)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 9)
    trainer = trainers.Trainer(model, optimizer, F.cross_entropy, scheduler=scheduler)
    trainer.run(loader, loader, 15, 11)
    assert trainer.step == 11 - 1
