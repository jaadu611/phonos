"""
AudioForge Universal Speech Specialist (Generalist of All Domains, Specialist of Human Speech).
Stabilized HiFi-GAN Architecture with Spectral Normalization and Rebalanced Complex STFT Phase Loss.
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console

from specialists.base_unet import HighResSpeechGenerator, MultiScaleSpeechDiscriminator

console = Console()


def apply_extreme_degradations(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Applies aggressive multi-domain acoustic stress:
    - Non-linear overdrive saturation (drive 1.5x - 7.5x)
    - Granular impulse crackle, electrical buzz & diaphragm grit
    - Tactical bit-crush quantization (6-10 bits)
    - Random spectral tilt & pink/brown atmospheric noise
    """
    out = waveform.copy()

    # 1. Non-linear Overdrive Saturation (75% probability)
    if random.random() < 0.75:
        drive = random.uniform(1.8, 7.5)
        out = np.tanh(out * drive) + 0.08 * np.sin(np.pi * np.clip(out * drive, -1.0, 1.0))
        peak = np.max(np.abs(out)) + 1e-8
        out = (out / peak) * 0.88

    # 2. Granular Impulse Crackle & Static (65% probability)
    if random.random() < 0.65:
        crackle_density = random.uniform(0.001, 0.006)
        mask = np.random.rand(len(out)) < crackle_density
        grains = (np.random.rand(len(out)) * 2 - 1) * random.uniform(0.10, 0.35)
        out[mask] += grains[mask]

    # 3. Bit-Crushing (50% probability)
    if random.random() < 0.50:
        bits = random.choice([6, 8, 10])
        steps = 2 ** bits
        out = np.round(out * steps) / steps

    # 4. Pink/Brown Background Grain (60% probability)
    if random.random() < 0.60:
        noise = np.random.randn(len(out))
        grain = np.cumsum(noise)
        grain = grain - np.mean(grain)
        grain = grain / (np.max(np.abs(grain)) + 1e-8) * random.uniform(0.02, 0.12)
        out += grain

    return np.clip(out, -0.98, 0.98)


class UniversalSpeechGANTrainer:
    """
    Stabilized Universal Speech Specialist GAN Trainer (HiFi-GAN Style).
    """

    def __init__(
        self,
        hours: float = 14.0,
        batch_size: int = 4,
        base_channels: int = 48,
        max_kept_checkpoints: int = 2,
    ):
        self.hours = hours
        self.batch_size = batch_size
        self.base_channels = base_channels
        self.max_kept_checkpoints = max_kept_checkpoints
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.speech_dir = Path("training_data/clean_speech")
        self.noise_base = Path("training_data/universal_noise")

        self.specialist_dir = Path("specialists/universal_speech")
        self.checkpoint_dir = self.specialist_dir / "checkpoints"
        self.test_audio_dir = Path("examples/test_audio")

        self.specialist_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.test_audio_dir.mkdir(parents=True, exist_ok=True)

        self.saved_checkpoints: List[Path] = []
        self.saved_test_wavs: List[Path] = []

        self.sr = 16000
        self.segment_duration = 2.0
        self.n_fft = 512
        self.hop_length = 160

        # Generator (~37.0M params) + Spectral-Normalized Discriminator (~8.6M params)
        self.generator = HighResSpeechGenerator(base_channels=self.base_channels).to(self.device)
        self.discriminator = MultiScaleSpeechDiscriminator().to(self.device)

        # Optimizers (Balanced Learning Rates: Generator lr=2e-4, Discriminator lr=5e-5)
        self.opt_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999),
            weight_decay=1e-5,
        )
        self.opt_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=0.00005,  # 4x lower LR to prevent discriminator overpower
            betas=(0.5, 0.999),
            weight_decay=1e-5,
        )

        total_steps = int(hours * 3600 * 10)
        self.sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_g, T_max=total_steps, eta_min=1e-5)
        self.sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_d, T_max=total_steps, eta_min=5e-6)

        self.scaler_g = torch.amp.GradScaler('cuda', enabled=(self.device == "cuda"))
        self.scaler_d = torch.amp.GradScaler('cuda', enabled=(self.device == "cuda"))

        self.step = 0
        self.best_loss = float("inf")

        self.load_all_files()

        g_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        d_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
        total_p = g_params + d_params

        console.print(f"[bold blue]AudioForge Stabilized Universal Speech GAN Trainer[/bold blue]")
        console.print(f"Generator: HighResSpeechGenerator  | Params: {g_params:,} (~{g_params*4/1024/1024:.2f} MB)")
        console.print(f"Discriminator: MultiScalePatchGAN (Spectral Norm) | Params: {d_params:,} (~{d_params*4/1024/1024:.2f} MB)")
        console.print(f"Total Pipeline: {total_p:,} (~{total_p*4/1024/1024:.2f} MB) | AMP: Active | Batch: {self.batch_size}\n")

    def load_all_files(self):
        self.speech_files = list(self.speech_dir.rglob("*.flac")) + list(self.speech_dir.rglob("*.wav"))
        console.print(f"  Clean Speech Pool : {len(self.speech_files)} files")

        self.noise_files: Dict[str, List[Path]] = {}
        for cat_dir in sorted(self.noise_base.iterdir()):
            if cat_dir.is_dir():
                files = list(cat_dir.glob("*.wav")) + list(cat_dir.glob("*.flac")) + list(cat_dir.glob("*.mp3")) + list(cat_dir.glob("*.ogg"))
                if files:
                    self.noise_files[cat_dir.name] = files

        total_noise_files = sum(len(f) for f in self.noise_files.values())
        console.print(f"  Universal Noise Pool: {len(self.noise_files)} categories ({total_noise_files:,} audio files)\n")

    def load_audio_segment(self, path: Path) -> np.ndarray:
        samples_needed = int(self.sr * self.segment_duration)
        try:
            audio, _ = librosa.load(str(path), sr=self.sr, mono=True)
        except Exception:
            return np.zeros(samples_needed, dtype=np.float32)

        if len(audio) < samples_needed:
            audio = np.pad(audio, (0, samples_needed - len(audio)))

        start = random.randint(0, len(audio) - samples_needed)
        return audio[start : start + samples_needed]

    def generate_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        samples = int(self.sr * self.segment_duration)

        # 1. Clean speech
        clean_speech = self.load_audio_segment(random.choice(self.speech_files))
        peak_speech = np.max(np.abs(clean_speech)) + 1e-8
        clean_speech = (clean_speech / peak_speech) * random.uniform(0.60, 0.85)

        # 2. Pick 2 to 6 random noise categories
        num_noises = random.randint(2, min(6, len(self.noise_files)))
        active_cats = random.sample(list(self.noise_files.keys()), num_noises)

        noise_layers = []
        for cat in active_cats:
            n_file = random.choice(self.noise_files[cat])
            n_wav = self.load_audio_segment(n_file)
            n_peak = np.max(np.abs(n_wav)) + 1e-8
            gain = 10.0 ** (random.uniform(-10.0, 6.0) / 20.0)
            noise_layers.append((n_wav / n_peak) * 0.6 * gain)

        total_noise = sum(noise_layers) if noise_layers else np.zeros(samples, dtype=np.float32)

        raw_mixture = clean_speech + total_noise
        degraded_mixture = apply_extreme_degradations(raw_mixture, sr=self.sr)

        mix_peak = np.max(np.abs(degraded_mixture))
        if mix_peak > 0.95:
            scale = 0.95 / mix_peak
            degraded_mixture *= scale
            clean_speech *= scale

        mix_stft = librosa.stft(degraded_mixture.astype(np.float32), n_fft=self.n_fft, hop_length=self.hop_length)
        tgt_stft = librosa.stft(clean_speech.astype(np.float32), n_fft=self.n_fft, hop_length=self.hop_length)

        return (
            np.stack([mix_stft.real, mix_stft.imag], axis=0),
            np.stack([tgt_stft.real, tgt_stft.imag], axis=0),
        )

    def generate_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_x, batch_y = [], []
        for _ in range(self.batch_size):
            x, y = self.generate_pair()
            batch_x.append(x)
            batch_y.append(y)

        return (
            torch.tensor(np.array(batch_x), dtype=torch.float32),
            torch.tensor(np.array(batch_y), dtype=torch.float32),
        )

    def clean_old_artifacts(self):
        while len(self.saved_checkpoints) > self.max_kept_checkpoints:
            old_ckpt = self.saved_checkpoints.pop(0)
            if old_ckpt.exists() and "best" not in old_ckpt.name:
                try:
                    old_ckpt.unlink()
                except Exception:
                    pass

        while len(self.saved_test_wavs) > 4:
            old_wav = self.saved_test_wavs.pop(0)
            if old_wav.exists():
                try:
                    old_wav.unlink()
                except Exception:
                    pass

    def train(self):
        console.print(f"[bold green]Starting {self.hours:.1f}-Hour Stabilized Universal GAN Training[/bold green]\n")

        start_time = time.time()
        target_seconds = self.hours * 3600.0

        while time.time() - start_time < target_seconds:
            batch_x, batch_y = self.generate_batch()
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # -------------------------------------------------------------
            # 1. TRAIN DISCRIMINATOR (Every 2nd Step)
            # -------------------------------------------------------------
            loss_d = torch.tensor(0.0, device=self.device)
            if self.step % 2 == 0:
                self.opt_d.zero_grad()
                with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                    with torch.no_grad():
                        fake_y = self.generator(batch_x)

                    scores_real, _ = self.discriminator(batch_y)
                    loss_d_real = sum(torch.mean((s - 1.0) ** 2) for s in scores_real) / len(scores_real)

                    scores_fake, _ = self.discriminator(fake_y.detach())
                    loss_d_fake = sum(torch.mean(s ** 2) for s in scores_fake) / len(scores_fake)

                    loss_d = 0.5 * (loss_d_real + loss_d_fake)

                self.scaler_d.scale(loss_d).backward()
                self.scaler_d.unscale_(self.opt_d)
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5.0)
                self.scaler_d.step(self.opt_d)
                self.scaler_d.update()
                self.sched_d.step()

            # -------------------------------------------------------------
            # 2. TRAIN GENERATOR (Every Step, Complex STFT Phase Leads)
            # -------------------------------------------------------------
            self.opt_g.zero_grad()
            with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                fake_y = self.generator(batch_x)

                # A. Complex STFT Phase & Harmonic Coherence (Weight: 1.00)
                recon_loss = F.l1_loss(fake_y, batch_y)

                # B. Magnitude Envelope (Weight: 0.30)
                pred_mag = torch.sqrt(fake_y[:, 0] ** 2 + fake_y[:, 1] ** 2 + 1e-8)
                tgt_mag = torch.sqrt(batch_y[:, 0] ** 2 + batch_y[:, 1] ** 2 + 1e-8)
                mag_loss = F.l1_loss(pred_mag, tgt_mag)

                # C. High-Frequency Consonant Clarity (>1.5 kHz) (Weight: 0.20)
                hf_loss = F.l1_loss(pred_mag[:, 24:], tgt_mag[:, 24:])

                # D. Adversarial Loss (Weight: 0.50 - Strong Signal!)
                scores_gen, feats_fake = self.discriminator(fake_y)
                loss_adv = sum(torch.mean((s - 1.0) ** 2) for s in scores_gen) / len(scores_gen)

                # E. Feature Matching Stabilizer (Weight: 0.30)
                with torch.no_grad():
                    _, feats_real = self.discriminator(batch_y)

                loss_fm = 0.0
                for f_fake_scale, f_real_scale in zip(feats_fake, feats_real):
                    for f_f, f_r in zip(f_fake_scale, f_real_scale):
                        loss_fm += F.l1_loss(f_f, f_r)
                loss_fm = loss_fm / (len(feats_fake) * len(feats_fake[0]))

                # Rebalanced Combined Generator Loss
                loss_g = (
                    1.00 * recon_loss +
                    0.30 * mag_loss +
                    0.20 * hf_loss +
                    0.50 * loss_adv +
                    0.30 * loss_fm
                )

            self.scaler_g.scale(loss_g).backward()
            self.scaler_g.unscale_(self.opt_g)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5.0)
            self.scaler_g.step(self.opt_g)
            self.scaler_g.update()
            self.sched_g.step()

            if loss_g.item() < self.best_loss:
                self.best_loss = loss_g.item()
                self.save_checkpoint(is_best=True)

            elapsed = time.time() - start_time
            self.step += 1

            if self.step % 200 == 0 or self.step == 1:
                remaining_h = max(0.0, (target_seconds - elapsed) / 3600.0)
                lr = self.sched_g.get_last_lr()[0]
                console.print(
                    f"  Step [bold cyan]{self.step:6d}[/bold cyan] | "
                    f"Gen: [bold yellow]{loss_g.item():.4f}[/bold yellow] | "
                    f"Recon: {recon_loss.item():.4f} | "
                    f"Adv: {loss_adv.item():.4f} | "
                    f"Disc: {loss_d.item():.4f} | "
                    f"LR: {lr:.6f} | "
                    f"Elapsed: {elapsed/3600:.2f}h | "
                    f"Remaining: {remaining_h:.2f}h"
                )

            if self.step % 1000 == 0:
                self.save_checkpoint(is_best=False)
                self.test_speech_extraction()
                self.clean_old_artifacts()

        self.save_checkpoint(is_best=True)
        self.test_speech_extraction()
        self.clean_old_artifacts()
        console.print(f"\n[bold green]✓ Stabilized GAN Training Complete![/bold green] Steps: {self.step:,} | Best Loss: {self.best_loss:.5f}")

    def save_checkpoint(self, is_best: bool = False):
        filename = f"universal_speech_gan_step{self.step}.pt" if not is_best else "universal_speech_specialist_best.pt"
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "step": self.step,
                "generator_state_dict": self.generator.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "loss": self.best_loss,
                "base_channels": self.base_channels,
                "domain": "universal_speech_all_domains",
            },
            str(path),
        )
        if not is_best:
            self.saved_checkpoints.append(path)
            console.print(f"  [green]✓ Saved Checkpoint:[/green] {path}")

    def test_speech_extraction(self):
        """
        High-Fidelity Contiguous Inference (Zero Overlap-Add Flanger Distortion).
        """
        test_file = Path("examples/test_audio/long_combat_mission_benchmark.wav")
        if not test_file.exists():
            test_file = Path("examples/test_audio/podcast_battlefield_mix.wav")
        if not test_file.exists():
            return

        from core.clean_inference import extract_clean_speech
        out_path = self.test_audio_dir / f"universal_speech_step{self.step}.wav"

        extract_clean_speech(
            model=self.generator,
            audio_input=test_file,
            output_path=out_path,
            sr=self.sr,
            segment_duration=self.segment_duration,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            device=self.device,
        )
        self.saved_test_wavs.append(out_path)
        console.print(f"  [green]✓ Exported Test Audio (Pure Contiguous):[/green] [bold]{out_path}[/bold]")

        self.generator.train()


if __name__ == "__main__":
    trainer = UniversalSpeechGANTrainer(hours=14.0, batch_size=4, base_channels=48, max_kept_checkpoints=2)
    trainer.train()
