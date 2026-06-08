import pytest
import torch

from splifft import core
from splifft import types as t


@pytest.mark.parametrize(
    ("chunk_size", "hop_size", "target_num_samples"),
    [
        (16, 8, 40),  # 50%
        (24, 24, 120),  # 0%
        (32, 12, 64),  # fractional
        (16, 1, 20),  # extreme to test normalization
        (16, 8, 32),  # perfect hop
        (32, 16, 137),
    ],
)
def test_streaming_stitcher_matches_stitch_chunks(
    chunk_size: int,
    hop_size: int,
    target_num_samples: int,
) -> None:
    torch.manual_seed(13)
    num_stems = 3
    num_channels = 2
    batch_size = 4

    original_audio = t.RawAudioTensor(torch.randn(num_channels, target_num_samples))

    chunk_generator = core.generate_chunks(
        original_audio,
        chunk_size=t.ChunkSize(chunk_size),
        hop_size=t.HopSize(hop_size),
        batch_size=t.BatchSize(batch_size),
        padding_mode="reflect",
    )

    processed_chunks = []
    for batch in chunk_generator:
        mock_output = batch.unsqueeze(1).expand(-1, num_stems, -1, -1).clone()
        processed_chunks.append(t.SeparatedChunkedTensor(mock_output))

    window = t.WindowTensor(core._get_window_fn("hann", chunk_size, torch.device("cpu")))

    legacy = core.stitch_chunks(
        processed_chunks=processed_chunks,
        num_stems=t.NumModelStems(num_stems),
        chunk_size=t.ChunkSize(chunk_size),
        hop_size=t.HopSize(hop_size),
        target_num_samples=t.Samples(target_num_samples),
        window=window,
    )

    stitcher = core.StreamingStitcher(
        chunk_size=t.ChunkSize(chunk_size),
        hop_size=t.HopSize(hop_size),
        target_num_samples=t.Samples(target_num_samples),
        window=window,
    )

    blocks: list[torch.Tensor] = []
    for batch in processed_chunks:
        for chunk in batch.unbind(0):
            block = stitcher.step(chunk)
            if block is not None:
                blocks.append(block)

    tail = stitcher.flush()
    if tail is not None:
        blocks.append(tail)

    assert blocks
    streamed = torch.cat(blocks, dim=-1)
    assert streamed.shape == legacy.shape
    torch.testing.assert_close(streamed, legacy, rtol=0.0, atol=0.0)
