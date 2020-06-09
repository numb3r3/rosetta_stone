import pytest
from rosetta.utils.containers import TensorMap, TensorTuple, AverageMeter, AverageDictMeter
import torch


def test_map():
    map = TensorMap(a=1, b=2)
    map["c"] = 3
    for k, v in map.items():
        assert map[k] == getattr(map, k)

    for k in ["update", "keys", "items", "values", "clear", "copy", "get", "pop"]:
        with pytest.raises(KeyError):
            setattr(map, k, 1)


def test_tensortuple():
    a = torch.randn(3, 3), torch.randn(3, 3)
    t = TensorTuple(a)
    assert t[0].dtype == torch.float32

    assert t.to(torch.int32)[0].dtype == torch.int32


def test_average_meter():
    avg_meter = AverageMeter()
    avg_meter.update(2)
    assert avg_meter.avg == 2

    avg_meter.update(2, n=3)
    assert avg_meter.avg == 2

    avg_meter.update(12)
    assert avg_meter.avg == 4

def test_average_dict_meter():
    avg_dict_meter = AverageDictMeter()
    avg_dict_meter.update({"a": 1, "b": 2, "c": 3})
    assert avg_dict_meter["a"].avg == 1

    avg_dict_meter.update({"a": 3, "b": 3, "c": 1.2})
    assert avg_dict_meter["a"].avg == 2
    assert avg_dict_meter["b"].avg == 2.5
    assert avg_dict_meter["c"].avg == 2.1

    assert len(avg_dict_meter) == 3

