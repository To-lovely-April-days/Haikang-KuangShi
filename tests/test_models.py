"""骨干构建、离线权重、checkpoint 往返。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from oresorter.ml import models as M


@pytest.mark.parametrize("backbone", M.BACKBONES)
def test_build_forward(backbone):
    m = M.build_model(backbone, 7, pretrained=True)
    m.eval()
    with torch.no_grad():
        y = m(torch.zeros(2, 3, 160, 160))
    assert y.shape == (2, 7)
    assert M.algo_name(backbone) == f"cnn-{backbone}"


def test_missing_weights_falls_back(tmp_path):
    m = M.build_model("resnet18", 3, pretrained=True, weights_dir=tmp_path)
    assert m.fc.out_features == 3


def test_freeze_backbone_and_head_params():
    m = M.build_model("resnet18", 5, pretrained=False)
    M.freeze_backbone(m, "resnet18", True)
    trainable = M.count_parameters(m, trainable_only=True)
    assert trainable == sum(p.numel() for p in M.head_parameters(m, "resnet18"))
    M.freeze_backbone(m, "resnet18", False)
    assert M.count_parameters(m, trainable_only=True) == M.count_parameters(m)


def test_checkpoint_roundtrip(tmp_path):
    m = M.build_model("mobilenet_v3_large", 4, pretrained=False)
    meta = {"backbone": "mobilenet_v3_large", "classes": ["A", "B", "C", "D"], "image_size": 128}
    path = tmp_path / "train" / "best.pt"
    M.save_checkpoint(path, m, meta)
    assert path.exists() and not list(path.parent.glob(".*.tmp"))
    m2, meta2 = M.load_checkpoint(path)
    assert meta2 == meta and not m2.training
    x = torch.randn(1, 3, 128, 128)
    m.eval()
    with torch.no_grad():
        assert np.allclose(m(x).numpy(), m2(x).numpy(), atol=1e-6)
    with pytest.raises(ValueError):
        M.save_checkpoint(path, m, {"backbone": "resnet18"})


def test_select_device_auto():
    d = M.select_device("auto")
    assert d.type in {"cpu", "cuda", "mps"}
    assert M.select_device("cpu").type == "cpu"
