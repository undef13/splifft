# %%
import sys
from functools import lru_cache
from pathlib import Path

import torch

from splifft.models.bs_roformer import BSRoformer, BSRoformerParams
from splifft.utils.mil import BlobFile, parse_mil_program, resolve_path

PATH_BASE = Path(__file__).parent.parent
PATH_RAW = PATH_BASE / "models" / "roformer.mlmodelc"
FP_MIL = PATH_RAW / "model.mil"
FP_WEIGHTS = PATH_RAW / "weights" / "weights.bin"


with open(FP_MIL, "r") as f:
    mil_content = f.read()

program = parse_mil_program(mil_content)

# %% torch preprocessing

torch.set_default_dtype(torch.float16)
model_cfg = BSRoformerParams(
    dim=256,
    depth=12,
    stereo=True,
    output_stem_names=tuple(f"_{i}" for i in range(6)),
    use_shared_bias=True,
    time_transformer_depth=1,
    freq_transformer_depth=1,
    chunk_size=588800,
    stft_hop_length=512,
)
model = BSRoformer(model_cfg)

alias = {
    "cos_emb_time": "op_1429_to_fp16",  # (1151, 64)
    "sin_emb_time": "op_1473_to_fp16",  # (1151, 64)
    "cos_emb_freq": "op_1760_to_fp16",  # (64, 64)
    "sin_emb_freq": "op_1804_to_fp16",  # (64, 64)
    "shared_qkv_bias": "linear_62_bias_0_to_fp16",  # (1536,)
    "shared_out_bias": "linear_64_bias_0_to_fp16",
}


# torch: mask_estimators.5.to_freqs.40.0.0.weight
# MIL: mask_estimators_5_to_freqs_40_0_0_weight_to_fp16
# MIL -> torch map
def torch_name_to_mil_op_name(name: str) -> str:
    for torch_name, mil_op_name in alias.items():
        if torch_name == name:
            return mil_op_name
    return f"{name.replace('.', '_')}_to_fp16"


mil_name_to_torch: dict[str, tuple[str, torch.nn.Parameter]] = {}  # mil -> torch
with torch.no_grad():
    for name_ext, param_ext in model.named_parameters():
        param_ext.copy_(torch.full_like(param_ext, torch.nan))
        mil_op_name = torch_name_to_mil_op_name(name_ext)
        mil_name_to_torch[mil_op_name] = (name_ext, param_ext)
# %% mil preprocessing

mil_to_torch_dtype = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


@lru_cache(maxsize=128)
def load_weights(path: Path) -> bytes:
    with open(path, "rb") as f:
        data = f.read()
    return data


def is_normal(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all() and not torch.isnan(tensor).any())


names_torch_loaded: list[str] = []
for op in program.functions[0].operations:
    result = mil_name_to_torch.get(op.op_name, None)
    if result is None:
        status = "F" if op.value is not None else " "
        # continue
    else:
        status = "."
        name_ext, param_ext = result
        torch_shape = param_ext.shape
        torch_dtype = param_ext.dtype

        assert op.value is not None
        mil_shape = op.value.shape
        mil_dtype_str = op.value.dtype

        if tuple(torch_shape) != tuple(mil_shape):
            print(
                f"{op.op_name} shape mismatch: "
                f"torch: {tuple(torch_shape)} != mil: {tuple(mil_shape)}",
                file=sys.stderr,
            )
        mil_dtype_torch = mil_to_torch_dtype.get(mil_dtype_str.lower(), None)
        if torch_dtype != mil_dtype_torch:
            print(
                f"{op.op_name} dtype mismatch: "
                f"torch: {torch_dtype} != mil: {mil_dtype_str} (expected {mil_dtype_torch})",
                file=sys.stderr,
            )

        mil_val = op.value.value
        if isinstance(mil_val, BlobFile):
            assert mil_val.path and mil_val.offset
            weights = load_weights(resolve_path(mil_val.path, PATH_RAW))
            weight = torch.frombuffer(
                weights,
                dtype=torch_dtype,
                count=param_ext.numel(),
                offset=mil_val.offset + 64,
            ).reshape(param_ext.shape)
        else:
            weight = torch.tensor(mil_val, dtype=param_ext.dtype).reshape(param_ext.shape)
        with torch.no_grad():
            param_ext.copy_(weight)
        names_torch_loaded.append(name_ext)

    # print(f"{status:2} {op.dbg(compact=True)}")

print("# model.named_parameters()")
for name_ext, param_ext in model.named_parameters():
    status = "F" if not is_normal(param_ext) else "."
    if status == ".":
        continue
    print(f"{status:2} {name_ext:50} {param_ext.shape} {param_ext.dtype}")


model_weights = model.state_dict()
torch.save(model_weights, PATH_MODELS / "roformer.pt")
# %%
# check against frazer's weights
model_ext = torch.load(PATH_MODELS / "roformer_frazer.pt", map_location="cpu", weights_only=True)
# raise SystemExit
# %%
for name_ext, param_ext in model_ext.items():
    for torch_name, mil_op_name in alias.items():
        if mil_op_name.removesuffix("_to_fp16") == name_ext:
            name_ext = torch_name  # old weights still use mil params, override
    if "rotary_embed" in name_ext:
        status = "s"  # we use a custom impl
    elif name_ext not in model_weights:
        status = "F"
    else:
        status = "."
    if torch.allclose(param_ext, torch.zeros_like(param_ext), atol=1e-11):
        status += "(all zeros)"
    # print(f"{status:2} {name_ext:50} {param_ext.shape} {param_ext.dtype}")
    if status == "s":
        continue
# %%
