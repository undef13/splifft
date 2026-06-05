import pytest
import torch
from dataclasses import dataclass
from typing import Literal

from splifft.models.utils.attend import Attend

try:
    import sageattention  # noqa: F401

    HAS_SAGE = True
except ImportError:
    HAS_SAGE = False

HAS_CUDA = torch.cuda.is_available()


def naiive_attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    b, h, seq_q, d = q.shape
    _, _, seq_k, _ = k.shape

    scale = d**-0.5
    sim = torch.einsum("bhid,bhjd->bhij", q.float(), k.float()) * scale

    if is_causal:
        causal_mask = torch.ones((seq_q, seq_k), device=q.device, dtype=torch.bool).triu_(
            diagonal=seq_k - seq_q + 1
        )
        sim.masked_fill_(causal_mask, torch.finfo(sim.dtype).min)

    attn = sim.softmax(dim=-1)
    out = torch.einsum("bhij,bhjd->bhid", attn, v.float()).to(q.dtype)
    return out


@dataclass
class AttendTestCase:
    name: str
    b: int
    h: int
    seq_q: int
    seq_k: int
    d: int
    is_causal: bool
    backend: Literal["sdpa", "sage"]


TEST_CASES = [
    AttendTestCase("sdpa_standard_aligned", 2, 4, 128, 128, 64, False, "sdpa"),
    AttendTestCase("sdpa_causal_aligned", 2, 4, 128, 128, 64, True, "sdpa"),
    AttendTestCase("sdpa_cross_unaligned", 1, 2, 256, 128, 32, False, "sdpa"),
]

if HAS_SAGE and HAS_CUDA:
    TEST_CASES.extend(
        [
            AttendTestCase("sage_standard_aligned", 2, 4, 128, 128, 64, False, "sage"),
            AttendTestCase("sage_causal_aligned", 2, 4, 128, 128, 64, True, "sage"),
        ]
    )

# we dont test for cross attention (seq_q != seq_k) with is_causal=True because
# pytorch sdpa causal mask alignment for unequal lengths is version-dependent
# and we dont use causal cross-attention anyway.


@pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c.name)
def test_attention_backends_parity(case: AttendTestCase) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.float16 if case.backend == "sage" else torch.float32

    torch.manual_seed(42)
    q = torch.randn(case.b, case.h, case.seq_q, case.d, device=device, dtype=dtype)
    k = torch.randn(case.b, case.h, case.seq_k, case.d, device=device, dtype=dtype)
    v = torch.randn(case.b, case.h, case.seq_k, case.d, device=device, dtype=dtype)

    attend = Attend(dropout=0.0, backend=case.backend).to(device)

    out = attend(q, k, v, is_causal=case.is_causal)
    ref_out = naiive_attend(q, k, v, is_causal=case.is_causal)

    atol, rtol = (3e-2, 3e-2) if case.backend == "sage" else (1e-6, 1e-6)
    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)
