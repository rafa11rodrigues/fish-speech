import os
import shutil
import time
from pathlib import Path

import click
import torch
from loguru import logger
from torchao.quantization import (
    float8_weight_only,
    int4_weight_only,
    int8_weight_only,
    quantize_,
)

from fish_speech.models.text2semantic.llama import DualARTransformer


def copy_metadata_and_symlink_codec(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    files_to_copy = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]
    for filename in files_to_copy:
        src_file = src_dir / filename
        if src_file.exists():
            shutil.copy2(src_file, dst_dir / filename)

    codec_src = src_dir / "codec.pth"
    codec_dst = dst_dir / "codec.pth"
    if codec_src.exists() and not codec_dst.exists():
        rel_path = Path(os.path.relpath(codec_src, dst_dir))
        codec_dst.symlink_to(rel_path)


@click.command()
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=Path, exists=True),
    default="checkpoints/s2-pro",
    help="Path to source checkpoint (e.g. checkpoints/s2-pro)",
)
@click.option(
    "--mode",
    type=click.Choice(["fp8", "int8", "int4"]),
    default="fp8",
    help="TorchAO quantization mode (fp8, int8, int4)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Destination directory for quantized model.",
)
def quantize_torchao(checkpoint_path: Path, mode: str, output_dir: Path | None):
    t0 = time.time()
    logger.info(f"Loading model from {checkpoint_path}...")
    model = DualARTransformer.from_pretrained(str(checkpoint_path), load_weights=True)

    if output_dir is None:
        output_dir = Path(f"checkpoints/s2-pro-torchao-{mode}")

    logger.info(f"Applying TorchAO {mode} quantization...")
    if mode == "fp8":
        quantize_(model, float8_weight_only())
    elif mode == "int8":
        quantize_(model, int8_weight_only())
    elif mode == "int4":
        target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Packing INT4 weights on {target_device}...")
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                mod.to(target_device)
                quantize_(mod, int4_weight_only(group_size=128))

    copy_metadata_and_symlink_codec(checkpoint_path, output_dir)
    quantize_path = output_dir / "model.pth"

    logger.info(f"Saving quantized weights to {quantize_path}...")
    quantize_path.unlink(missing_ok=True)
    torch.save(model.state_dict(), quantize_path)
    logger.info(
        f"TorchAO {mode} quantization complete! Took {time.time() - t0:.2f}s. Saved to {output_dir}"
    )


if __name__ == "__main__":
    quantize_torchao()
