import pytest
from rosetta.core import lr_schedulers, optimizers, trainers
import torch


class DumyModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)
        self.loss_fct = torch.nn.functional.cross_entropy

    def forward(self, x, y):
        logits = self.layer(x)
        loss = self.loss_fct(logits, y.view(-1))
        output = {}
        metrics = {}
        return output, loss, metrics


def test_basic_trainer():
    model = DumyModel()
    optimizer = optimizers.SGD()
    scheduler = lr_schedulers.StepLR(9)
    trainer = trainers.Trainer(model, optimizer, lr_scheduler=scheduler)

    loader = [(torch.randn(2, 10), torch.zeros(2, dtype=torch.long)) for _ in range(10)]

    trainer.train(loader)
    assert pytest.approx(trainer.optimizer.param_groups[0]["lr"], 0.01)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    trainer = trainers.Trainer(model, optimizer, scheduler=scheduler)
    trainer.train(loader)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 9)
    trainer = trainers.Trainer(model, optimizer, scheduler=scheduler)
    trainer.run(loader, loader, 15, 11)
    assert trainer.global_step == 11 - 1
