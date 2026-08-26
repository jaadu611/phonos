"""
AudioForge 37.0M Flagship GAN Training Runner (10,000 Steps).
Features:
- 37.0M Dilated Residual Generator (base_channels=48)
- 8.6M Spectral-Normalized Multi-Scale PatchGAN Discriminator
- 59 Universal Sound Categories Ingested into RAM Cache
- Multi-Scale Complex STFT + Mag + HF Formants + Anti-Noise Floor + Adv Loss
- Pure Contiguous Evaluation (Zero Overlap-Add Flanger Distortion)
"""

import os
import random
import time
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console

from specialists.base_unet import HighResSpeechGenerator, MultiScaleSpeechDiscriminator
from core.clean_inference import extract_clean_speech

console = Console()


def preload_audio(file_paths: List[Path], sr: int = 16000) -> List[np.ndarray]:
    cache = []
    for p in file_paths:
        try:
            w, _ = librosa.load(str(p), sr=sr, mono=True)
            if len(w) >= int(sr * 1.0):
                cache.append(w.astype(np.float32))
        except Exception:
            pass
    return cache


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sr = 16000
    segment_duration = 2.0
    samples_needed = int(sr * segment_duration)
    n_fft = 512
    hop_length = 160
    target_steps = 10000
    batch_size = 4

    console.print("\n[bold magenta]===========================================================[/bold magenta]")
    console.print("[bold cyan]AudioForge 37.0M Flagship GAN Specialist Training (10K Steps)[/bold cyan]")
    console.print(f"Device: {device} | Target Steps: {target_steps} | Batch Size: {batch_size}")
    console.print("[bold magenta]===========================================================[/bold magenta]\n")

    # 1. Preload Datasets
    speech_dir = Path("training_data/clean_speech")
    noise_dir = Path("training_data/universal_noise")

    speech_files = list(speech_dir.rglob("*.flac")) + list(speech_dir.rglob("*.wav"))
    console.print(f"[yellow]Preloading {len(speech_files)} speech files into RAM...[/yellow]")
    speech_cache = preload_audio(speech_files, sr=sr)
    console.print(f"  [green]✓ Clean Speech Cached:[/green] {len(speech_cache)} files")

    console.print(f"[yellow]Preloading 59 Universal Noise Categories into RAM...[/yellow]")
    noise_cache: Dict[str, List[np.ndarray]] = {}
    for d in sorted(noise_dir.iterdir()):
        if d.is_dir():
            files = list(d.glob("*.wav")) + list(d.glob("*.flac")) + list(d.glob("*.mp3")) + list(d.glob("*.ogg"))
            if files:
                loaded = preload_audio(files, sr=sr)
                if loaded:
                    noise_cache[d.name] = loaded

    total_noise = sum(len(v) for v in noise_cache.values())
    console.print(f"  [green]✓ Universal Noise Cached:[/green] {len(noise_cache)} categories ({total_noise} files)")

    # 2. Instantiate 37.0M Generator & 8.6M Discriminator
    generator = HighResSpeechGenerator(base_channels=48).to(device)
    discriminator = MultiScaleSpeechDiscriminator().to(device)

    g_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    console.print(f"\n[bold green]Generator Parameters:[/bold green]     {g_params:,} (~{g_params*4/1024/1024:.2f} MB)")
    console.print(f"[bold green]Discriminator Parameters:[/bold green] {d_params:,} (~{d_params*4/1024/1024:.2f} MB)")
    console.print(f"[bold green]Total Model Capacity:[/bold green]     {g_params+d_params:,} (~{(g_params+d_params)*4/1024/1024:.2f} MB)\n")

    # 3. Optimizers & Schedulers
    opt_g = torch.optim.AdamW(generator.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-5)

    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=target_steps, eta_min=1e-5)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=target_steps, eta_min=5e-6)

    scaler_g = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
    scaler_d = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    def get_slice(arr: np.ndarray) -> np.ndarray:
        if len(arr) < samples_needed:
            return np.pad(arr, (0, samples_needed - len(arr)))
        st = random.randint(0, len(arr) - samples_needed)
        return arr[st : st + samples_needed]

    def make_batch():
        bx, by = [], []
        for _ in range(batch_size):
            # Clean speech
            clean = get_slice(random.choice(speech_cache))
            clean = clean / (np.max(np.abs(clean)) + 1e-8) * random.uniform(0.65, 0.85)

            # Noise mixture (1-4 stems)
            num_n = random.randint(1, min(4, len(noise_cache)))
            active_cats = random.sample(list(noise_cache.keys()), num_n)
            noise_layers = []
            for cat in active_cats:
                n_wav = get_slice(random.choice(noise_cache[cat]))
                n_peak = np.max(np.abs(n_wav)) + 1e-8
                gain = 10.0 ** (random.uniform(-8.0, 4.0) / 20.0)
                noise_layers.append((n_wav / n_peak) * 0.6 * gain)

            tot_noise = sum(noise_layers) if noise_layers else np.zeros(samples_needed, dtype=np.float32)
            mix = clean + tot_noise
            p = np.max(np.abs(mix))
            if p > 0.95:
                scale = 0.92 / p
                mix = mix * scale
                clean = clean * scale

            stft_m = librosa.stft(mix, n_fft=n_fft, hop_length=hop_length)
            stft_s = librosa.stft(clean, n_fft=n_fft, hop_length=hop_length)
            bx.append(np.stack([stft_m.real, stft_m.imag]))
            by.append(np.stack([stft_s.real, stft_s.imag]))

        return (
            torch.tensor(np.array(bx), dtype=torch.float32).to(device),
            torch.tensor(np.array(by), dtype=torch.float32).to(device),
        )

    # Training Loop
    start_time = time.time()
    best_loss = float("inf")
    ckpt_dir = Path("specialists/universal_speech/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold green]Starting 10,000-Step Adversarial Training Loop...[/bold green]\n")

    for step in range(1, target_steps + 1):
        batch_x, batch_y = make_batch()

        # A. Train Discriminator (Every 2nd step)
        loss_d = torch.tensor(0.0, device=device)
        if step % 2 == 0:
            opt_d.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                with torch.no_grad():
                    fake_y = generator(batch_x)

                scores_real, _ = discriminator(batch_y)
                loss_d_real = sum(torch.mean((s - 1.0) ** 2) for s in scores_real) / len(scores_real)

                scores_fake, _ = discriminator(fake_y.detach())
                loss_d_fake = sum(torch.mean(s ** 2) for s in scores_fake) / len(scores_fake)

                loss_d = 0.5 * (loss_d_real + loss_d_fake)

            scaler_d.scale(loss_d).backward()
            scaler_d.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
            scaler_d.step(opt_d)
            scaler_d.update()
            sched_d.step()

        # B. Train Generator (Every Step)
        opt_g.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device == "cuda")):
            fake_y = generator(batch_x)

            # 1. Complex STFT Reconstruction
            recon_loss = F.l1_loss(fake_y, batch_y)

            # 2. Magnitude Envelope
            pred_mag = torch.sqrt(fake_y[:, 0] ** 2 + fake_y[:, 1] ** 2 + 1e-8)
            tgt_mag = torch.sqrt(batch_y[:, 0] ** 2 + batch_y[:, 1] ** 2 + 1e-8)
            mag_loss = F.l1_loss(pred_mag, tgt_mag)

            # 3. High-Frequency Formants (>1.5 kHz)
            hf_loss = F.l1_loss(pred_mag[:, 24:], tgt_mag[:, 24:])

            # 4. Anti-Noise Floor Penalty (Suppresses background noise)
            silent_bins = tgt_mag < 0.015
            anti_noise = pred_mag[silent_bins].mean() * 4.0 if silent_bins.any() else torch.tensor(0.0, device=device)

            # 5. Adversarial Loss (Generator wants Discriminator to output 1.0)
            scores_gen, feats_fake = discriminator(fake_y)
            loss_adv = sum(torch.mean((s - 1.0) ** 2) for s in scores_gen) / len(scores_gen)

            # 6. Feature Matching Loss
            with torch.no_grad():
                _, feats_real = discriminator(batch_y)
            loss_fm = 0.0
            for scale_idx in range(len(feats_fake)):
                for layer_idx in range(len(feats_fake[scale_idx])):
                    loss_fm += F.l1_loss(feats_fake[scale_idx][layer_idx], feats_real[scale_idx][layer_idx])
            loss_fm = loss_fm / max(1, sum(len(f) for f in feats_fake))

            # Total Generator Loss
            loss_g = (
                1.00 * recon_loss
                + 0.40 * mag_loss
                + 0.25 * hf_loss
                + 0.35 * anti_noise
                + 0.10 * loss_adv
                + 0.10 * loss_fm
            )

        scaler_g.scale(loss_g).backward()
        scaler_g.unscale_(opt_g)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
        scaler_g.step(opt_g)
        scaler_g.update()
        sched_g.step()

        if loss_g.item() < best_loss:
            best_loss = loss_g.item()

        if step % 500 == 0 or step == 1:
            elapsed = time.time() - start_time
            rate = step / elapsed
            console.print(
                f"  Step {step:5d}/{target_steps} | "
                f"Gen: [bold yellow]{loss_g.item():.4f}[/bold yellow] | "
                f"Disc: [bold cyan]{loss_d.item():.4f}[/bold cyan] | "
                f"Recon: {recon_loss.item():.4f} | "
                f"NoisePen: {anti_noise.item():.4f} | "
                f"Elapsed: {elapsed:.1f}s ({rate:.1f} st/s)"
            )

    # 4. Save Final 10K Checkpoint
    ckpt_path = ckpt_dir / "universal_speech_37m_gan_10k.pt"
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "loss": best_loss,
            "step": target_steps,
            "base_channels": 48,
        },
        str(ckpt_path),
    )
    console.print(f"\n[bold green]✓ Checkpoint Saved:[/bold green] {ckpt_path}")

    # 5. Run Pure Contiguous Inference on Podcast Benchmark
    podcast_mix = "examples/test_audio/podcast_battlefield_mix.wav"
    output_wav = "examples/test_audio/podcast_battlefield_37m_gan_10k.wav"

    console.print(f"[yellow]Exporting restored audio from {podcast_mix}...[/yellow]")
    extract_clean_speech(
        model=generator,
        audio_input=podcast_mix,
        output_path=output_wav,
        sr=sr,
        segment_duration=segment_duration,
        n_fft=n_fft,
        hop_length=hop_length,
        device=device,
    )
    console.print(f"[bold green]✓ Exported Final Restored Audio:[/bold green] [bold cyan]{output_wav}[/bold cyan]\n")


if __name__ == "__main__":
    main()
