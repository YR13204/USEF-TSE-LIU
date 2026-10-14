"""Score a LibriMix USEF checkpoint without changing the running trainer."""

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from hyperpyyaml import load_hyperpyyaml
from mir_eval.separation import bss_eval_sources


def read_manifest(path):
    entries = {}
    for line in path.read_text().splitlines():
        key, audio_path, *rest = line.split()
        entries[key] = (Path(audio_path), int(rest[0]) if rest else None)
    return entries


def read_audio(path, sample_rate):
    audio, actual_rate = sf.read(path, dtype="float32")
    if actual_rate != sample_rate or audio.ndim != 1:
        raise ValueError(f"Expected mono {sample_rate} Hz audio: {path}")
    return audio


def si_snr(estimate, reference):
    estimate = estimate.astype(np.float64) - np.mean(estimate)
    reference = reference.astype(np.float64) - np.mean(reference)
    projection = np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-8)
    target = projection * reference
    residual = estimate - target
    return 10 * np.log10((np.dot(target, target) + 1e-8) / (np.dot(residual, residual) + 1e-8))


def bss_sdr(estimate, reference):
    sdr, _, _, _ = bss_eval_sources(
        reference[np.newaxis, :], estimate[np.newaxis, :], compute_permutation=False
    )
    return float(sdr[0])


def predict(model, mixture, auxiliary, chunk_samples, use_bf16, device):
    aux = torch.from_numpy(auxiliary).unsqueeze(0).to(device)
    parts = []
    with torch.inference_mode():
        for offset in range(0, len(mixture), chunk_samples):
            chunk = mixture[offset : offset + chunk_samples]
            padded = np.pad(chunk, (0, chunk_samples - len(chunk)))
            inp = torch.from_numpy(padded).unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                output = model(inp, aux)
            parts.append(output[0, : len(chunk)].float().cpu().numpy())
    return np.concatenate(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, help="Defaults to test_path in config")
    parser.add_argument("--subset", choices=("all", "Libri2Mix", "Libri3Mix"), default="all")
    parser.add_argument("--limit", type=int, default=20, help="0 scores the full selected test set")
    parser.add_argument("--max-seconds", type=float, default=0, help="0 scores full utterances")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open() as handle:
        config = load_hyperpyyaml(handle)
    sample_rate = config["sample_rate"]
    test_path = args.test_path or Path(config["test_path"])
    mix = read_manifest(test_path / config["mix_scp"])
    ref = read_manifest(test_path / config["ref_scp"])
    aux = read_manifest(test_path / config["aux_scp"])
    if mix.keys() != ref.keys() or mix.keys() != aux.keys():
        raise ValueError("mix, ref and aux manifests must have matching IDs")

    model = config["modules"]["masknet"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {key.removeprefix("module."): value for key, value in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state)
    model.to(args.device).eval()
    use_bf16 = args.device == "cuda" and config.get("use_bf16", False)
    chunk_samples = int(config.get("valid_chunk_seconds", 4) * sample_rate)
    aux_samples = int(config.get("valid_aux_seconds", 3) * sample_rate)
    if chunk_samples <= 0:
        raise ValueError("valid_chunk_seconds must be positive for bounded inference")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("id", "seconds", "sdr_mix", "sdr_est", "sdri", "sisnr_mix", "sisnr_est", "sisnri")
    rows = []
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, (mix_path, _) in mix.items():
            if args.subset != "all" and args.subset not in mix_path.parts:
                continue
            mixture = read_audio(mix_path, sample_rate)
            reference = read_audio(ref[key][0], sample_rate)
            auxiliary = read_audio(aux[key][0], sample_rate)
            if aux[key][1] is not None:
                auxiliary = auxiliary[: aux[key][1]]
            auxiliary = auxiliary[:aux_samples]
            if args.max_seconds:
                cap = int(args.max_seconds * sample_rate)
                mixture = mixture[:cap]
                reference = reference[:cap]
            length = min(len(mixture), len(reference))
            if length == 0 or len(auxiliary) == 0:
                raise ValueError(f"Empty audio for {key}")
            mixture = mixture[:length]
            reference = reference[:length]
            estimate = predict(model, mixture, auxiliary, chunk_samples, use_bf16, args.device)[:length]
            sdr_mix = bss_sdr(mixture, reference)
            sdr_est = bss_sdr(estimate, reference)
            sisnr_mix = si_snr(mixture, reference)
            sisnr_est = si_snr(estimate, reference)
            row = dict(
                id=key,
                seconds=length / sample_rate,
                sdr_mix=sdr_mix,
                sdr_est=sdr_est,
                sdri=sdr_est - sdr_mix,
                sisnr_mix=sisnr_mix,
                sisnr_est=sisnr_est,
                sisnri=sisnr_est - sisnr_mix,
            )
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            print(f"{len(rows)} {key}: SDR {sdr_est:.2f}, SDRi {row['sdri']:.2f}, SI-SNRi {row['sisnri']:.2f}", flush=True)
            if args.limit and len(rows) >= args.limit:
                break
    if not rows:
        raise ValueError(f"No examples found for subset {args.subset}")
    means = {name: np.mean([row[name] for row in rows]) for name in fields[2:]}
    print(f"Checkpoint epoch {checkpoint['epoch']}; {len(rows)} examples; results: {args.output}")
    print("Mean " + ", ".join(f"{name}={value:.3f}" for name, value in means.items()))


if __name__ == "__main__":
    main()
