#!/usr/bin/env python3
"""
Phonos: Command line interface for 44.1kHz speech restoration and enhancement.
Uses VoiceFixer with mixed NF4/FP8 quantization and a harmonic presence exciter.
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf


def main():
    """Parse command line arguments and execute requested audio processing command."""
    if len(sys.argv) > 1 and sys.argv[1] not in ["enhance", "info", "-h", "--help"]:
        sys.argv.insert(1, "enhance")

    parser = argparse.ArgumentParser(
        prog="phonos",
        description="Phonos: Universal 44.1kHz Speech Restoration & Enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py recording.wav -o restored.wav
  python cli.py enhance recording.wav --no-exciter -o restored.wav
  python cli.py info recording.wav
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    enhance_parser = subparsers.add_parser("enhance", help="Restore and enhance degraded speech")
    enhance_parser.add_argument("input", help="Input audio file path")
    enhance_parser.add_argument("--output", "-o", help="Output file path (default: <input>_restored.wav)")
    enhance_parser.add_argument(
        "--chunk-size",
        type=float,
        default=5.0,
        help="Chunk duration in seconds (default: 5.0, keeps peak VRAM < 950 MB)",
    )
    enhance_parser.add_argument(
        "--no-exciter",
        action="store_true",
        help="Disable harmonic presence exciter and output raw neural restoration",
    )

    info_parser = subparsers.add_parser("info", help="Show audio file metadata")
    info_parser.add_argument("input", help="Input audio file path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "enhance":
            cmd_enhance(args)
        elif args.command == "info":
            cmd_info(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        err_msg = str(e).strip() or type(e).__name__
        print(f"Error: {err_msg}", file=sys.stderr)
        sys.exit(1)


def safe_trim_center(est, ref):
    """Align estimation tensor duration to reference samples by center cropping or padding."""
    import torch

    target_len = ref.shape[-1]
    if est.shape[-1] > target_len:
        diff = est.shape[-1] - target_len
        left = diff // 2
        return est[..., left : left + target_len]
    elif est.shape[-1] < target_len:
        diff = target_len - est.shape[-1]
        left = diff // 2
        return torch.nn.functional.pad(est, (left, diff - left))
    return est


def apply_harmonic_exciter(
    wav: np.ndarray,
    sample_rate: int = 44100,
    cutoff_hz: float = 4000.0,
    drive: float = 1.5,
    blend: float = 0.30,
) -> np.ndarray:
    """
    Restore vocal presence and high-frequency air.
    Filters audio above cutoff_hz with a Butterworth highpass, applies soft tanh saturation
    to synthesize harmonic overtones, and blends the result back with peak limiting.
    """
    from scipy.signal import butter, sosfilt

    sos = butter(4, cutoff_hz, "highpass", fs=sample_rate, output="sos")
    highs = sosfilt(sos, wav)
    sat_highs = np.tanh(highs * drive)
    excited = wav + blend * sat_highs
    peak = np.max(np.abs(excited))
    if peak > 1.0:
        excited = (excited / peak) * 0.98
    return excited.astype(np.float32)


def restore_voicefixer_16bit(
    vf,
    input_path: Path,
    output_path: Path,
    cuda: bool = True,
    chunk_size: float = 5.0,
    exciter: bool = True,
):
    """
    Restore degraded audio in chunks using VoiceFixer with FP16 autocast.
    Applies neural denoising and dereverberation, vocoder harmonic synthesis,
    and optional high-frequency harmonic presence excitation.
    """
    import torch
    from voicefixer.restorer.model import from_log
    from voicefixer.tools.pytorch_util import tensor2numpy
    from voicefixer.tools.wav import save_wave

    with torch.no_grad():
        wav_10k = vf._load_wav(str(input_path), sample_rate=44100)
        if wav_10k.shape[0] == 0:
            raise ValueError(f"Input audio is empty: {input_path}")

        if cuda and torch.cuda.is_available():
            vf._model = vf._model.cuda()

        vf._model.eval()

        res = []
        seg_length = int(44100 * chunk_size)
        break_point = seg_length
        while break_point < wav_10k.shape[0] + seg_length:
            segment = wav_10k[break_point - seg_length : break_point]

            # Symmetrically pad segments under 2048 samples to satisfy STFT reflection bounds (n_fft // 2 = 1024)
            if len(segment) < 2048:
                total_pad = 2048 - len(segment)
                left_pad = total_pad // 2
                right_pad = total_pad - left_pad
                seg_input = np.pad(segment, (left_pad, right_pad))
            else:
                seg_input = segment

            sp, mel_noisy = vf._pre(vf._model, seg_input, cuda=cuda)
            del sp

            # Denoise and reconstruct spectrogram with FP16 autocast on CUDA
            gen = vf._model.generator
            with torch.amp.autocast("cuda", enabled=(cuda and torch.cuda.is_available())):
                clean = gen.denoiser(mel_noisy) * mel_noisy
                clean_log = torch.log10(torch.clamp(clean, min=1e-8))
                mel_log = torch.log10(torch.clamp(mel_noisy, min=1e-8))
                unet_in = torch.cat([mel_log, clean_log], dim=1)
                denoised_mel = from_log(gen.unet(unet_in)["mel"] + clean_log).float()
                del clean, clean_log, mel_log, unet_in

            # Synthesize 44.1kHz audio waveform using neural vocoder
            out = vf._model.vocoder(denoised_mel, cuda=cuda)
            del denoised_mel
            if torch.max(torch.abs(out)) > 1.0:
                out = out / torch.max(torch.abs(out))
            trimmed = safe_trim_center(out, segment)
            res.append(trimmed.cpu())
            del out, trimmed
            if cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()
            break_point += seg_length

        out = torch.cat(res, -1)
        np_wav = tensor2numpy(out.squeeze(0))
        if exciter:
            np_wav = apply_harmonic_exciter(np_wav, sample_rate=44100)
        save_wave(np_wav, fname=str(output_path), sample_rate=44100)


def download_checkpoint(dest_path: Path):
    """Download quantized model weights from Hugging Face or GitHub Releases if not present locally."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urls = [
        f"https://huggingface.co/jaadu611/phonos/resolve/main/{dest_path.name}",
        f"https://github.com/jaadu611/phonos/releases/download/v1.0/{dest_path.name}",
    ]
    import urllib.request

    print(f"Model checkpoint not found locally. Auto-downloading {dest_path.name}...")
    for url in urls:
        try:
            print(f"  Attempting source: {url}")

            def reporthook(count, block_size, total_size):
                if total_size > 0:
                    percent = min(100, int(count * block_size * 100 / total_size))
                    sys.stdout.write(
                        f"\r  Download Progress: {percent}% ({count * block_size / (1024**2):.1f}/{total_size / (1024**2):.1f} MB)"
                    )
                    sys.stdout.flush()

            urllib.request.urlretrieve(url, str(dest_path), reporthook=reporthook)
            print("\n  Download completed successfully.")
            return True
        except Exception as e:
            print(f"\n  Source unavailable ({e}). Trying fallback...")
            if dest_path.exists():
                dest_path.unlink()
    print(f"Error: Could not automatically download {dest_path.name}. Please place it in {dest_path.parent}")
    return False


def cmd_enhance(args):
    """Load model checkpoint into VoiceFixer and restore speech in the target file."""
    import torch

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_restored.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Pipeline: Universal 44.1kHz Speech Restoration (Denoise + Dereverb + Formant Synthesis)")

    print("Loading VoiceFixer (Universal 44.1kHz SOTA Restorer)...")
    from voicefixer import VoiceFixer

    vf = VoiceFixer()

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_mixed = ckpt_dir / "phonos_v1.0.pt"

    if not ckpt_mixed.exists():
        download_checkpoint(ckpt_mixed)

    if ckpt_mixed.exists():
        import bitsandbytes.functional as Fbnb

        # Safely deserialize bundle and inject dequantized weights into model modules
        torch.serialization.add_safe_globals([Fbnb.QuantState])
        bundle = torch.load(str(ckpt_mixed), map_location="cpu", weights_only=True)
        gen_weights = bundle["generator"]["weights"]
        gen_meta = bundle["generator"]["meta"]
        gen_sd = vf._model.generator.state_dict()
        for k, v in gen_weights.items():
            if k in gen_sd:
                m = gen_meta[k]
                if m["type"] == "nf4":
                    deq = Fbnb.dequantize_nf4(v.cuda() if cuda_available else v, m["state"])
                    if m["pad"] > 0:
                        deq = deq[:-m["pad"]]
                    gen_sd[k].copy_(deq.view(m["orig_shape"]).to(gen_sd[k].dtype))
                else:
                    scale = m["scale"]
                    gen_sd[k].copy_((v.float() / scale).to(gen_sd[k].dtype))
        vf._model.generator.load_state_dict(gen_sd)

        voc_bundle = bundle["vocoder"]
        voc_sd = vf._model.vocoder.state_dict()
        for k, v in voc_bundle["weights"].items():
            if k in voc_sd:
                scale = voc_bundle["scales"].get(k, 1.0)
                voc_sd[k].copy_((v.float() / scale).to(voc_sd[k].dtype))
        vf._model.vocoder.load_state_dict(voc_sd)
        ckpt_info = f"Loaded {ckpt_mixed.name} ({ckpt_mixed.stat().st_size / (1024**2):.1f} MB Ultra-Compact Mixed NF4/FP8)"

        # Release dequantization allocations to minimize VRAM usage before inference
        del bundle, gen_weights, gen_meta, gen_sd, voc_bundle, voc_sd
        if cuda_available:
            torch.cuda.empty_cache()
    else:
        ckpt_info = "Loaded default model weights"
    print(f"Model Checkpoint: {ckpt_info}")

    accel_note = " (16-bit Tensor Core Accelerated)" if cuda_available else ""
    print(f"Processing on {device_name}{accel_note}...")

    t0 = time.time()
    restore_voicefixer_16bit(
        vf=vf,
        input_path=input_path,
        output_path=output_path,
        cuda=cuda_available,
        chunk_size=getattr(args, "chunk_size", 5.0),
        exciter=not getattr(args, "no_exciter", False),
    )
    elapsed = time.time() - t0

    info = sf.info(str(output_path))
    speed = info.duration / elapsed if elapsed > 0 else 0.0
    print(f"Done in {elapsed:.2f}s ({speed:.1f}x real-time).")
    print(f"Restored 44.1kHz studio audio saved to: {output_path}")


def cmd_info(args):
    """Display audio file container metadata without importing deep learning frameworks."""
    path = Path(args.input)
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    info = sf.info(str(path))
    print(f"\nAudio File Metadata: {path.name}")
    print(f"  Duration:     {info.duration:.2f} seconds")
    print(f"  Sample Rate:  {info.samplerate} Hz")
    print(f"  Channels:     {info.channels}")
    print(f"  Format:       {info.format} ({info.subtype})")
    print(f"  Frames:       {info.frames:,}\n")


if __name__ == "__main__":
    main()
