import argparse
import copy
import csv
import hashlib
import json
import os
import pickle
import time
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import requests
import torch
import torchvision.transforms as transforms
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from modules.ca_adain import ca_adain
from modules.smsa import prepare_style_mixing_self_attention_features


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
feat_maps = []
DEFAULT_PRECOMPUTED_PATH = ""
DEFAULT_CHECKPOINT_URL = (
    "https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/"
    "resolve/main/sd-v1-4.ckpt"
)
REMOTE_CHECKPOINT_CACHE_ROOT = PROJECT_ROOT / "models" / "remote-checkpoints"
MINIMUM_CHECKPOINT_BYTES = 1024 * 1024
RUN_METADATA_FILENAME = "run_metadata.json"
PAIR_PROGRESS_FILENAME = "pair_progress.jsonl"
RUN_METADATA_SCHEMA_VERSION = 4
PHASE_PRESERVING_FOURIER_CORRECTION_SAMPLER_MODE = "dynamic_content"
STYLE_MIXING_SELF_ATTENTION_INTERNAL_MODE = "smsa_full"
RUN_METADATA_COMPAT_KEYS = (
    "project",
    "method",
    "cnt",
    "sty",
    "output_path",
    "precomputed",
    "num_pairs",
    "seed",
    "pfc_steps",
    "fusion_alpha",
    "fusion_beta",
    "lambda_amp",
    "style_weight",
    "content_weight",
    "gamma",
    "tau_s",
    "tau_c",
    "adain_eps",
    "model_config",
    "checkpoint_url",
    "ckpt",
)


def resolve_from_cwd(path_text):
    path = Path(path_text)
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def resolve_from_project(path_text):
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def is_usable_checkpoint(path):
    return (
        path.is_file()
        and path.stat().st_size >= MINIMUM_CHECKPOINT_BYTES
    )


def preserve_checkpoint_file(path, label):
    """Move an invalid or partial checkpoint aside without deleting it."""
    if not path.exists():
        return None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    target = path.with_name(f"{path.name}.{label}.{timestamp}")
    suffix = 1
    while target.exists():
        target = path.with_name(f"{path.name}.{label}.{timestamp}.{suffix}")
        suffix += 1
    path.replace(target)
    return target


def remote_checkpoint_cache_path(checkpoint_url):
    parsed = urlparse(checkpoint_url)
    filename = Path(unquote(parsed.path)).name or "model.ckpt"
    url_key = hashlib.sha256(checkpoint_url.encode("utf-8")).hexdigest()[:12]
    return REMOTE_CHECKPOINT_CACHE_ROOT / url_key / filename


def download_remote_checkpoint(checkpoint_url):
    """Download and cache the checkpoint declared by the remote URL."""
    if not checkpoint_url.startswith(("https://", "http://")):
        raise ValueError("--checkpoint_url must use an HTTP(S) URL.")

    destination = remote_checkpoint_cache_path(checkpoint_url)
    if is_usable_checkpoint(destination):
        print(f"Using cached remote checkpoint: {destination}")
        return destination
    if destination.exists():
        preserved = preserve_checkpoint_file(destination, "invalid")
        print(f"Preserved invalid remote checkpoint as: {preserved}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(f"{destination.name}.part")
    if partial_path.exists():
        preserved = preserve_checkpoint_file(partial_path, "incomplete")
        print(f"Preserved previous incomplete download as: {preserved}")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    headers = {"User-Agent": "StylePFC/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    print(f"Downloading the default checkpoint from: {checkpoint_url}")
    try:
        with requests.get(
            checkpoint_url,
            headers=headers,
            stream=True,
            allow_redirects=True,
            timeout=(15, 120),
        ) as response:
            response.raise_for_status()
            with partial_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
    except Exception:
        if partial_path.exists():
            print(f"Incomplete remote download retained at: {partial_path}")
        raise

    if not is_usable_checkpoint(partial_path):
        preserved = preserve_checkpoint_file(partial_path, "invalid")
        raise RuntimeError(
            "The remote response was not a usable checkpoint; it was preserved at "
            f"{preserved}."
        )
    partial_path.replace(destination)
    print(f"Remote checkpoint cached at: {destination}")
    return destination


def resolve_model_checkpoint(checkpoint_url, local_checkpoint_path):
    """Prefer the remote checkpoint and fall back to the declared local path."""
    try:
        return download_remote_checkpoint(checkpoint_url)
    except Exception as remote_error:
        print(f"Remote checkpoint unavailable: {remote_error}")
        if is_usable_checkpoint(local_checkpoint_path):
            print(f"Falling back to local checkpoint: {local_checkpoint_path}")
            return local_checkpoint_path
        raise RuntimeError(
            "Unable to obtain the Stable Diffusion v1.4 checkpoint from the "
            f"remote URL, and no usable local fallback exists at {local_checkpoint_path}."
        ) from remote_error


def parse_pfc_steps(text):
    text = text.strip()
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def list_image_names(directory):
    return [
        name
        for name in sorted(os.listdir(directory))
        if Path(name).suffix.lower() in IMAGE_SUFFIXES
    ]


def build_pair_plan(style_names, content_names):
    pairs = []
    for style_name in style_names:
        for content_name in content_names:
            pairs.append(
                {
                    "content_name": content_name,
                    "style_name": style_name,
                    "output_name": f"{Path(content_name).stem}_stylized_{Path(style_name).stem}.png",
                }
            )
    return pairs


def group_pairs_by_style(pairs):
    grouped = OrderedDict()
    for pair in pairs:
        grouped.setdefault(pair["style_name"], []).append(pair)
    return grouped


def write_pair_manifest(output_dir, pairs):
    manifest_path = output_dir / "pair_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "content_name", "style_name", "output_name"])
        writer.writeheader()
        for index, pair in enumerate(pairs):
            writer.writerow(
                {
                    "index": index,
                    "content_name": pair["content_name"],
                    "style_name": pair["style_name"],
                    "output_name": pair["output_name"],
                }
            )


def compute_run_signature(metadata):
    signature_payload = {key: metadata.get(key) for key in RUN_METADATA_COMPAT_KEYS}
    signature_json = json.dumps(signature_payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(signature_json.encode("utf-8")).hexdigest()


def load_existing_metadata(metadata_path):
    if not metadata_path.is_file():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def is_metadata_compatible(existing_metadata, current_metadata):
    if not isinstance(existing_metadata, dict):
        return False
    if existing_metadata.get("run_metadata_schema_version") != RUN_METADATA_SCHEMA_VERSION:
        return False
    return existing_metadata.get("run_signature") == compute_run_signature(
        current_metadata
    )


def is_valid_output_image(path):
    if not path.is_file():
        return False
    try:
        if path.stat().st_size <= 0:
            return False
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def collect_output_status(output_dir, pairs):
    completed_outputs = set()
    invalid_outputs = []
    for pair in pairs:
        output_file = output_dir / pair["output_name"]
        if is_valid_output_image(output_file):
            completed_outputs.add(pair["output_name"])
        elif output_file.exists():
            invalid_outputs.append(output_file.name)
    return completed_outputs, invalid_outputs


def append_pair_progress(progress_path, status, pair=None, detail=""):
    payload = {
        "timestamp": round(time.time(), 3),
        "status": status,
    }
    if pair is not None:
        payload.update(
            {
                "content_name": pair["content_name"],
                "style_name": pair["style_name"],
                "output_name": pair["output_name"],
            }
        )
    if detail:
        payload["detail"] = detail
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_run_metadata(output_dir, metadata):
    metadata_path = output_dir / RUN_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def save_img_from_sample(model, samples_ddim, filename):
    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
    x_sample = 255.0 * rearrange(x_image_torch[0].cpu().numpy(), "c h w -> h w c")
    Image.fromarray(x_sample.astype(np.uint8)).save(filename)


def quarantine_stale_part(path):
    """Move an interrupted temporary image aside instead of deleting it."""
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    quarantine_dir = path.parent / "quarantine" / stamp
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    target = quarantine_dir / path.name
    suffix = 1
    while target.exists():
        target = quarantine_dir / f"{path.stem}_{suffix}{path.suffix}"
        suffix += 1
    path.replace(target)
    return target


def load_img(path):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    print(f"Loaded input image of size ({width}, {height}) from {path}")
    image = transforms.CenterCrop(min(width, height))(image)
    image = image.resize((512, 512), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing and verbose:
        print("missing keys:")
        print(missing)
    if unexpected and verbose:
        print("unexpected keys:")
        print(unexpected)
    model.eval()
    return model


def build_run_metadata(opt, cnt_dir, sty_dir, output_dir, precomputed_dir, pairs, active_pfc_steps):
    metadata = {
        "run_metadata_schema_version": RUN_METADATA_SCHEMA_VERSION,
        "project": "StylePFC",
        "cnt": str(cnt_dir),
        "sty": str(sty_dir),
        "output_path": str(output_dir),
        "precomputed": str(precomputed_dir) if precomputed_dir else "",
        "num_pairs": len(pairs),
        "seed": opt.seed,
        "method": (
            "CA-AdaIN + Style-Mixing Self-Attention + "
            "Phase-Preserving Fourier Correction"
        ),
        "pfc_steps": active_pfc_steps,
        "fusion_alpha": opt.fusion_alpha,
        "fusion_beta": opt.fusion_beta,
        "lambda_amp": opt.lambda_amp,
        "style_weight": opt.style_weight,
        "content_weight": opt.content_weight,
        "gamma": opt.gamma,
        "tau_s": opt.tau_s,
        "tau_c": opt.tau_c,
        "adain_eps": opt.adain_eps,
        "model_config": str(opt.model_config),
        "checkpoint_url": opt.checkpoint_url,
        "ckpt": str(opt.ckpt),
    }
    return metadata


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full StylePFC pipeline: CA-AdaIN, Style-Mixing "
            "Self-Attention, and Phase-Preserving Fourier Correction."
        )
    )
    parser.add_argument("--cnt", default="./data/cnt", help="Content image directory.")
    parser.add_argument(
        "--sty",
        default="./data/sty",
        help="Style image directory or category root containing style subdirectories.",
    )
    parser.add_argument("--ddim_inv_steps", type=int, default=50, help="Number of DDIM inversion steps.")
    parser.add_argument("--save_feat_steps", type=int, default=50, help="Number of timesteps used to save inversion feature banks.")
    parser.add_argument("--start_step", type=int, default=49, help="Sampling restart step counted from the DDIM trajectory.")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM eta value used in sampling.")
    parser.add_argument("--H", type=int, default=512, help="Image height in pixel space.")
    parser.add_argument("--W", type=int, default=512, help="Image width in pixel space.")
    parser.add_argument("--C", type=int, default=4, help="Latent channel count.")
    parser.add_argument("--f", type=int, default=8, help="Latent downsampling factor.")
    parser.add_argument("--gamma", type=float, default=0.75, help="Query preservation weight.")
    parser.add_argument("--attn_layer", type=str, default="6,7,8,9,10,11", help="Comma-separated output block indices for attention injection.")
    parser.add_argument("--model_config", type=str, default="models/ldm/stable-diffusion-v1/v1-inference.yaml", help="Model config path relative to StylePFC root unless absolute.")
    parser.add_argument("--precomputed", type=str, default=DEFAULT_PRECOMPUTED_PATH, help=f"Cache directory for precomputed inversion feature banks. Default: {DEFAULT_PRECOMPUTED_PATH}.")
    parser.add_argument("--checkpoint_url", type=str, default=DEFAULT_CHECKPOINT_URL, help="Primary remote URL for the Stable Diffusion v1.4 checkpoint.")
    parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt", help="Local checkpoint fallback, relative to StylePFC root unless absolute.")
    parser.add_argument("--precision", type=str, default="autocast", help="Precision mode, usually 'autocast' or 'full'.")
    parser.add_argument("--output_path", type=str, default="output", help="Output directory for stylized images and metadata.")
    parser.add_argument("--seed", type=int, default=22, help="Random seed.")
    # Core method parameters.
    parser.add_argument("--fusion_alpha", type=float, default=0.2, help="PFC weight applied to the encoded Fourier candidate.")
    parser.add_argument("--fusion_beta", type=float, default=1.0, help="PFC weight applied to the original predicted-clean latent.")
    parser.add_argument( "--lambda_amp", type=float, default=0.5, help="Content-amplitude contribution in Phase-Preserving Fourier Correction.",)
    parser.add_argument("--pfc_steps", type=str, default="2,5,8", help="Comma-separated reverse-DDIM step indices where PFC is applied.")
    parser.add_argument("--style_weight", type=float, default=0.6, help="CA-AdaIN style statistic weight.")
    parser.add_argument("--content_weight", type=float, default=0.4, help="CA-AdaIN content statistic weight.")
    parser.add_argument("--tau_s", type=float, default=1.5, help="SMSA style-branch logit scale.")
    parser.add_argument("--tau_c", type=float, default=0.4, help="SMSA content-branch logit scale.")
    parser.add_argument("--adain_eps", type=float, default=1e-6, help="Numerical epsilon for attention-bank alignment.")
    return parser


def _run_single(opt):
    # Internal full-method routing settings (not public comparison switches).
    opt.smsa_mode = STYLE_MIXING_SELF_ATTENTION_INTERNAL_MODE
    opt.gamma_effective = opt.gamma
    opt.k_align_effective = True
    opt.v_align_effective = True
    opt.tau_s_effective = opt.tau_s
    opt.tau_c_effective = opt.tau_c

    cnt_dir = resolve_from_cwd(opt.cnt)
    sty_dir = resolve_from_cwd(opt.sty)
    output_dir = resolve_from_cwd(opt.output_path)
    precomputed_dir = resolve_from_cwd(opt.precomputed) if opt.precomputed else None
    model_config_path = resolve_from_project(opt.model_config)
    local_ckpt_path = resolve_from_project(opt.ckpt)

    pfc_steps = parse_pfc_steps(opt.pfc_steps)
    if any(step <= 0 or step > opt.save_feat_steps for step in pfc_steps):
        raise ValueError(
            "PFC step indices must be positive and no greater than --save_feat_steps."
        )
    active_pfc_steps = pfc_steps

    output_dir.mkdir(parents=True, exist_ok=True)
    if precomputed_dir is not None:
        precomputed_dir.mkdir(parents=True, exist_ok=True)

    style_names = list_image_names(sty_dir)
    content_names = list_image_names(cnt_dir)
    pairs = build_pair_plan(style_names, content_names)
    selected_style_names = style_names
    selected_content_names = content_names
    if not pairs:
        raise RuntimeError("No style/content pairs were found. Please check the input directories.")

    write_pair_manifest(output_dir, pairs)
    completed_outputs, invalid_outputs = collect_output_status(output_dir, pairs)
    metadata = build_run_metadata(opt, cnt_dir, sty_dir, output_dir, precomputed_dir, pairs, active_pfc_steps)
    metadata_path = output_dir / RUN_METADATA_FILENAME
    existing_metadata = load_existing_metadata(metadata_path)
    if completed_outputs and existing_metadata is None:
        raise RuntimeError(
            f"Found {len(completed_outputs)} completed outputs in {output_dir} but {RUN_METADATA_FILENAME} is missing or invalid. "
            "Cannot safely resume without compatible run metadata."
        )
    if completed_outputs and not is_metadata_compatible(existing_metadata, metadata):
        raise RuntimeError(
            f"Existing outputs in {output_dir} were generated with a different configuration. "
            "Use a new output path or clean the directory before rerunning."
        )
    metadata["run_signature"] = compute_run_signature(metadata)
    metadata["resume_enabled"] = True
    metadata["completed_pairs_detected"] = len(completed_outputs)
    metadata["pending_pairs_detected"] = len(pairs) - len(completed_outputs)
    write_run_metadata(output_dir, metadata)
    progress_path = output_dir / PAIR_PROGRESS_FILENAME
    append_pair_progress(
        progress_path,
        "resume_scan",
        detail=f"completed={len(completed_outputs)} pending={len(pairs) - len(completed_outputs)} invalid={len(invalid_outputs)}",
    )
    if invalid_outputs:
        print(f"Found {len(invalid_outputs)} invalid existing outputs; they will be regenerated.")
    pending_pairs = [pair for pair in pairs if pair["output_name"] not in completed_outputs]
    if not pending_pairs:
        print("All output images already exist for this configuration; nothing to generate.")
        return

    seed_everything(opt.seed)
    model_config = OmegaConf.load(str(model_config_path))
    checkpoint_path = resolve_model_checkpoint(opt.checkpoint_url, local_ckpt_path)
    model = load_model_from_config(model_config, str(checkpoint_path))

    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(",")))
    ddim_inversion_steps = opt.ddim_inv_steps
    save_feature_timesteps = ddim_steps = opt.save_feat_steps

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model

    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
    time_range = np.flip(sampler.ddim_timesteps)
    idx_time_dict = {}
    time_idx_dict = {}
    for i, timestep in enumerate(time_range):
        idx_time_dict[timestep] = i
        time_idx_dict[i] = timestep

    global feat_maps
    feat_maps = [
        {"config": {"gamma": opt.gamma, "T": opt.tau_s}}
        for _ in range(len(time_range))
    ]

    def ddim_sampler_callback(pred_x0, xt, timestep):
        save_feature_maps_callback(timestep)
        save_feature_map(xt, "z_enc", timestep)
        save_feature_map(pred_x0, "pred_x0", timestep)

    def save_feature_maps(blocks, timestep, feature_type="input_block"):
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])) and block_idx in self_attn_output_block_indices:
                q = block[1].transformer_blocks[0].attn1.q
                k = block[1].transformer_blocks[0].attn1.k
                v = block[1].transformer_blocks[0].attn1.v
                save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", timestep)
                save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", timestep)
                save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", timestep)

    def save_feature_maps_callback(timestep):
        save_feature_maps(unet_model.output_blocks, timestep, "output_block")

    def save_feature_map(feature_map, filename, timestep):
        global feat_maps
        cur_idx = idx_time_dict[timestep]
        feat_maps[cur_idx][filename] = feature_map

    start_step = opt.start_step
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    uc = model.get_learned_conditioning([""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    pairs_by_style = group_pairs_by_style(pending_pairs)

    print(
        f"Selected {len(selected_style_names)} style images, "
        f"{len(selected_content_names)} content images, {len(pairs)} total pairs."
    )
    print(
        "Full method: CA-AdaIN + Style-Mixing Self-Attention (SMSA) + "
        "Phase-Preserving Fourier Correction (PFC); "
        f"PFC active steps: {active_pfc_steps}"
    )
    print(f"Image-level resume: {len(completed_outputs)} completed, {len(pending_pairs)} pending.")

    begin = time.time()
    for style_name, style_pairs in pairs_by_style.items():
        style_path = sty_dir / style_name
        init_sty = load_img(str(style_path)).to(device)
        style_feat_name = precomputed_dir / f"{style_path.stem}_sty.pkl" if precomputed_dir else None

        if style_feat_name and style_feat_name.is_file():
            print("Precomputed style feature loading:", style_feat_name)
            with style_feat_name.open("rb") as handle:
                sty_feat = pickle.load(handle)
            sty_z_enc = torch.clone(sty_feat[0]["z_enc"])
        else:
            init_sty = model.get_first_stage_encoding(model.encode_first_stage(init_sty))
            sty_z_enc, _ = sampler.encode_ddim(
                init_sty.clone(),
                num_steps=ddim_inversion_steps,
                unconditional_conditioning=uc,
                end_step=time_idx_dict[ddim_inversion_steps - 1 - start_step],
                callback_ddim_timesteps=save_feature_timesteps,
                img_callback=ddim_sampler_callback,
            )
            sty_feat = copy.deepcopy(feat_maps)
            sty_z_enc = feat_maps[0]["z_enc"]

        for pair in style_pairs:
            content_name = pair["content_name"]
            content_path = cnt_dir / content_name
            output_file = output_dir / pair["output_name"]
            if is_valid_output_image(output_file):
                print(f"[SKIP] Already exists: {output_file}")
                append_pair_progress(progress_path, "skipped_existing", pair)
                continue
            init_cnt = load_img(str(content_path)).to(device)
            cnt_feat_name = precomputed_dir / f"{content_path.stem}_cnt.pkl" if precomputed_dir else None

            if cnt_feat_name and cnt_feat_name.is_file():
                print("Precomputed content feature loading:", cnt_feat_name)
                with cnt_feat_name.open("rb") as handle:
                    cnt_feat = pickle.load(handle)
                cnt_z_enc = torch.clone(cnt_feat[0]["z_enc"])
            else:
                init_cnt = model.get_first_stage_encoding(model.encode_first_stage(init_cnt))
                cnt_z_enc, _ = sampler.encode_ddim(
                    init_cnt.clone(),
                    num_steps=ddim_inversion_steps,
                    unconditional_conditioning=uc,
                    end_step=time_idx_dict[ddim_inversion_steps - 1 - start_step],
                    callback_ddim_timesteps=save_feature_timesteps,
                    img_callback=ddim_sampler_callback,
                )
                cnt_feat = copy.deepcopy(feat_maps)
                cnt_z_enc = feat_maps[0]["z_enc"]

            missing_content = [
                index
                for index, feature_map in enumerate(cnt_feat)
                if "pred_x0" not in feature_map
            ]
            if missing_content:
                raise RuntimeError(
                    "Phase-Preserving Fourier Correction requires content inversion "
                    "feature banks "
                    "containing predicted-clean tensors at every timestep; "
                    f"missing indices: {missing_content}. Use a new/empty "
                    "--precomputed directory or omit --precomputed to regenerate it."
                )
            content_clean_latents_seq = [
                feature_map["pred_x0"] for feature_map in cnt_feat
            ]

            with torch.no_grad():
                with precision_scope("cuda"):
                    with model.ema_scope():
                        print(f"Inversion end: {time.time() - begin:.2f}s")
                        adain_z_enc = ca_adain(
                            cnt_z_enc,
                            sty_z_enc,
                            content_weight=opt.content_weight,
                            style_weight=opt.style_weight,
                        )

                        # Style-Mixing Self-Attention (SMSA) jointly normalizes
                        # style and aligned-content attention branches.
                        merged_features = prepare_style_mixing_self_attention_features(
                            opt,
                            cnt_feat,
                            sty_feat,
                            num_steps=len(cnt_feat),
                            start_step=start_step,
                        )

                        samples_ddim, _ = sampler.sample(
                            S=ddim_steps,
                            batch_size=1,
                            shape=shape,
                            verbose=False,
                            unconditional_conditioning=uc,
                            eta=opt.ddim_eta,
                            x_T=adain_z_enc,
                            injected_features=merged_features,
                            start_step=start_step,
                            content_clean_latents=content_clean_latents_seq,
                            style_clean_latents=None,
                            content_reference_image=None,
                            fpd_mode=PHASE_PRESERVING_FOURIER_CORRECTION_SAMPLER_MODE,
                            fusion_alpha=opt.fusion_alpha,
                            fusion_beta=opt.fusion_beta,
                            lambda_amp=opt.lambda_amp,
                            fpd_steps=active_pfc_steps,
                        )

                        tmp_output_file = output_file.with_name(f"{output_file.stem}.part{output_file.suffix}")
                        quarantine_stale_part(tmp_output_file)
                        save_img_from_sample(model, samples_ddim, tmp_output_file)
                        tmp_output_file.replace(output_file)
                        append_pair_progress(progress_path, "completed", pair)
                        del samples_ddim, merged_features
                        torch.cuda.empty_cache()

                        if precomputed_dir is not None:
                            if cnt_feat_name and not cnt_feat_name.is_file():
                                with cnt_feat_name.open("wb") as handle:
                                    pickle.dump(cnt_feat, handle)
                            if style_feat_name and not style_feat_name.is_file():
                                with style_feat_name.open("wb") as handle:
                                    pickle.dump(sty_feat, handle)

    print(f"Total end: {time.time() - begin:.2f}s")


def main():
    parser = build_argparser()
    opt = parser.parse_args()

    # A flat style directory keeps the original single-run behavior. When the
    # supplied style root contains category subdirectories (the default
    # data/sty layout), run every category independently and isolate outputs
    # and feature caches by category.
    sty_root = resolve_from_cwd(opt.sty)
    direct_style_images = list_image_names(sty_root)
    category_dirs = sorted(
        child
        for child in sty_root.iterdir()
        if child.is_dir() and list_image_names(child)
    ) if sty_root.is_dir() else []

    if not direct_style_images and category_dirs:
        output_root = resolve_from_cwd(opt.output_path)
        precomputed_root = resolve_from_cwd(opt.precomputed) if opt.precomputed else None
        print(
            f"Detected {len(category_dirs)} style categories under {sty_root}; "
            "running them independently."
        )
        for category_dir in category_dirs:
            category_opt = copy.copy(opt)
            category_opt.sty = str(category_dir)
            category_opt.output_path = str(output_root / category_dir.name)
            if precomputed_root is not None:
                category_opt.precomputed = str(precomputed_root / category_dir.name)
            print(f"\n===== StylePFC category: {category_dir.name} =====")
            _run_single(category_opt)
        return

    _run_single(opt)


if __name__ == "__main__":
    main()
