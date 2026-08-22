r"""OpenVision2 — image -> caption demo (full generative model).

Loads a released OpenVision2 vision encoder + its caption text decoder (both live in
the same `*-vision-only` HF repo) and captions an image. The image resolution is read
automatically from the encoder config, so you only pass the repo and the image.

    python caption.py --repo UCSC-VLAA/openvision2-vit-large-patch14-224-vision-only --image cat.jpg

Repos (all now include the caption decoder):
    openvision2-vit-large-patch14-224 / -336
    openvision2-vit-huge-patch14-224 / -336 / -448
    openvision2-vit-so400m-patch14-384
    openvision2-vit-giant-patch14-224
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# OpenVision2's customized open_clip (upstream pip open_clip is NOT compatible)
from src.convert_upload.open_clip.factory import create_vision_encoder_and_transforms
from src.convert_upload.modeling_openvision2_decoder import (
    OpenVision2TextDecoder,
    OpenVision2TextDecoderConfig,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_VOCAB = os.path.join(os.path.dirname(__file__), "assets", "bert_base_vocab_bos_eos.txt")


def preprocess(path, res):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = res / min(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = im.size
    l, t = (w - res) // 2, (h - res) // 2
    im = im.crop((l, t, l + res, t + res))
    x = (np.asarray(im, np.float32) - IMAGENET_MEAN) / IMAGENET_STD
    return torch.tensor(x.transpose(2, 0, 1)[None], dtype=torch.float32)  # [1,C,H,W]


def detokenize(ids, vocab, eos_id=2, specials=(0, 1, 2)):
    words = []
    for i in ids:
        if i == eos_id:
            break
        if i in specials:
            continue
        tok = vocab[i] if 0 <= i < len(vocab) else "[UNK]"
        if tok.startswith("##"):
            words[-1] = words[-1] + tok[2:] if words else tok[2:]
        else:
            words.append(tok)
    return " ".join(words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="UCSC-VLAA/openvision2-vit-large-patch14-224-vision-only")
    ap.add_argument("--image", required=True)
    ap.add_argument("--vocab", default=DEFAULT_VOCAB)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    # --- vision encoder (resolution taken from its config) ---
    enc = create_vision_encoder_and_transforms(
        model_name=f"hf-hub:{args.repo}", cache_dir=args.cache_dir).to(device).eval()
    vcfg = json.load(open(hf_hub_download(args.repo, "open_clip_config.json", cache_dir=args.cache_dir)))
    res = int(vcfg["model_cfg"]["vision_cfg"]["image_size"])

    # --- caption decoder ---
    dcfg = json.load(open(hf_hub_download(args.repo, "text_decoder_config.json", cache_dir=args.cache_dir)))
    dec = OpenVision2TextDecoder(OpenVision2TextDecoderConfig(
        width=dcfg["width"], depth=dcfg["depth"], num_heads=dcfg["num_heads"], mlp_dim=dcfg["mlp_dim"],
        vocab_size=dcfg["vocab_size"], vision_width=dcfg["vision_width"]))
    dec.load_state_dict(load_file(hf_hub_download(
        args.repo, "caption_decoder.safetensors", cache_dir=args.cache_dir)))
    dec = dec.to(device).eval()
    vocab = [l.rstrip("\n") for l in open(args.vocab)]

    # --- image -> caption ---
    with torch.no_grad():
        _, patch_tokens = enc(preprocess(args.image, res).to(device))  # [1, N, vision_width]
        ids = dec.generate(patch_tokens, max_len=args.max_len,
                           bos_id=dcfg["bos_id"], eos_id=dcfg["eos_id"])[0].tolist()
    print(detokenize(ids, vocab, eos_id=dcfg["eos_id"]))


if __name__ == "__main__":
    main()
