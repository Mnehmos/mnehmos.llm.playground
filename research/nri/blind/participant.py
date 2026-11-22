"""Standalone participant interface. It only reads the files in the released packet."""
from hashlib import sha256
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from nri import dimtask
from nri.model import TinyTyped


def new_model():
    return TinyTyped("blind", n_types=1, aux_classes=dimtask.N_ROLES,
                     vocab=dimtask.VOCAB, seq_len=dimtask.SEQ_LEN, n_out=dimtask.P).eval()


def load_case(directory=None):
    root = Path(directory) if directory is not None else Path(__file__).parent
    info = json.loads((root / "manifest.json").read_text())
    for name in ("checkpoint", "calibration"):
        digest = sha256((root / f"{name}.safetensors").read_bytes()).hexdigest()
        if digest != info[f"{name}_sha256"]:
            raise ValueError(f"{name} digest does not match this case")
    model = new_model()
    model.load_state_dict(load_file(root / "checkpoint.safetensors"))
    return model, load_file(root / "calibration.safetensors"), info


def save_edit(path, original, edited, info, target):
    """Export replacement values for changed parameters only, against an untouched original."""
    if target not in ("M", "A"):
        raise ValueError("target must be M or A")
    source, result = original.state_dict(), edited.state_dict()
    if source.keys() != result.keys():
        raise ValueError("the checkpoint architecture changed")
    replacements = {}
    allowed = dict(original.named_parameters())
    for name, before in source.items():
        after = result[name].detach().cpu()
        if before.shape != after.shape or before.dtype != after.dtype or not after.isfinite().all():
            raise ValueError(f"invalid tensor: {name}")
        if not torch.equal(before.detach().cpu(), after):
            if name not in allowed:
                raise ValueError(f"cannot edit buffer: {name}")
            replacements[name] = after.contiguous()
    save_file(replacements, path, metadata={"checkpoint_sha256": info["checkpoint_sha256"],
                                            "target": target})


if __name__ == "__main__":
    model, calibration, info = load_case()
    tokens = calibration["tokens"][:4]
    with torch.no_grad():
        print("Checkpoint", info["checkpoint_sha256"])
        print("Calibration rows", len(calibration["tokens"]))
        print("First four model predictions", model(tokens, torch.zeros_like(tokens)).argmax(1).tolist())
