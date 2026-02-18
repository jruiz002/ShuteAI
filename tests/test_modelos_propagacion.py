import pytest
import torch

from shuteai.propagacion.modelos import (
    ForecasterLSTM,
    PatchTransformer,
    PropagadorLSTM,
    PropagadorTransformer,
    PropMLP,
)

B, T, D = 3, 10, 32


@pytest.mark.parametrize("cls_modelo", [PropMLP, PropagadorLSTM, PropagadorTransformer])
def test_modo_a_forma_y_monotonia(cls_modelo):
    kw = {"d_txt": D} if cls_modelo is PropMLP else {"d_txt": D, "h": 16, "heads": 2}
    m = cls_modelo(**kw).eval()
    mask = torch.ones(B, T)
    mask[:, 7:] = 0
    curva = m(torch.randn(B, D), torch.randn(B, T, D), mask)
    assert curva.shape == (B, 96)
    assert torch.all(torch.diff(curva, dim=1) >= 0)


def test_lstm_ignora_tokens_de_relleno():
    m = PropagadorLSTM(d_txt=D, h=16, heads=2, dropout=0.0).eval()
    cls, tok = torch.randn(1, D), torch.randn(1, T, D)
    mask = torch.ones(1, T)
    mask[:, 6:] = 0
    tok2 = tok.clone()
    tok2[:, 6:] += 100.0  # cambiar solo el relleno no debe afectar
    assert torch.allclose(m(cls, tok, mask), m(cls, tok2, mask), atol=1e-5)


def test_scheduled_sampling_corre():
    m = PropagadorLSTM(d_txt=D, h=16, heads=2).train()
    y = torch.rand(B, 96).cumsum(1)
    out = m(torch.randn(B, D), torch.randn(B, T, D), torch.ones(B, T), y_real=y, p_real=0.5)
    out.sum().backward()


@pytest.mark.parametrize(
    "m",
    [
        ForecasterLSTM(12, 84, h=16),
        PatchTransformer(12, 84, 4, 2, h=16),
        PatchTransformer(12, 84, 1, 1, h=16),
    ],
)
def test_modo_b_continua_desde_ultimo_valor(m):
    m.eval()
    ctx = torch.rand(B, 12).cumsum(1)
    fut = m(ctx)
    assert fut.shape == (B, 84)
    assert torch.all(fut[:, 0] >= ctx[:, -1])
    assert torch.all(torch.diff(fut, dim=1) >= 0)
