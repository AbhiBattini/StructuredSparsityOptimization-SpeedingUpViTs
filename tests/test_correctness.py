from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from kernels.naive_sparse import pack_2to4
from sparsity.check_2to4 import is_2to4_compliant
from sparsity.make_2to4 import apply_2to4


def test_apply_2to4_is_exact():
    w = torch.randn(32, 64)
    sparse, mask = apply_2to4(w)

    ok_mask, _, _ = is_2to4_compliant(mask)
    assert ok_mask

    ok_sparse, _, _ = is_2to4_compliant(sparse)
    assert ok_sparse

    assert torch.allclose(sparse, w * mask)
    assert mask.sum().item() == w.numel() // 2


def test_2to4_rejects_bad_shape():
    bad = torch.randn(8, 10)
    ok, _, _ = is_2to4_compliant(bad)
    assert not ok


def test_pack_2to4_layout():
    w = torch.randn(16, 64)
    sparse, _ = apply_2to4(w)
    packed = pack_2to4(sparse)
    assert packed.values.shape == (16, 16, 2)
    assert packed.indices.shape == (16, 16, 2)
    assert packed.indices.dtype == torch.int32
