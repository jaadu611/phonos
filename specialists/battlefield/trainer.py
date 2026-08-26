"""
AudioForge Flagship Generative Adversarial Speech Specialist Trainer.
Trains a high-capacity (~41.3M parameter) Generator + Multi-Scale PatchGAN Discriminator with Feature Matching loss.
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


class BattlefieldGANTrainer:
    """
    Flagship Adversarial Speech Specialist Trainer.
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
        self.noise_base = Path("training_data/battlefield")

        self.specialist_dir = Path("specialists/battlefield")
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

        # Generator (~37M params) + Discriminator (~4.3M params)
        self.generator = HighResSpeechGenerator(base_channels=self.base_channels).to(self.device)
        self.discriminator = MultiScaleSpeechDiscriminator().to(self.device)

        # Separate Optimizers
        self.opt_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999),
            weight_decay=1e-5,
        )
        self.opt_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999),
            weight_decay=1e-5,
        )

        total_steps = int(hours * 3600 * 10)
        self.sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_g, T_max=total_steps, eta_min=1e-5)
        self.sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_d, T_max=total_steps, eta_min=1e-5)

        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == "cuda"))

        self.step = 0
        self.best_loss = float("inf")

        self.load_all_files()

        g_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        d_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
        total_p = g_params + d_params

        console.print(f"[bold blue]AudioForge Flagship GAN Speech Specialist Trainer[/bold blue]")
        console.print(f"Generator: HighResSpeechGenerator  | Params: {g_params:,} (~{g_params*4/1024/1024:.2f} MB)")
        console.print(f"Discriminator: MultiScalePatchGAN  | Params: {d_params:,} (~{d_params*4/1024/1024:.2f} MB)")
        console.print(f"Total Pipeline: {total_p:,} (~{total_p*4/1024/1024:.2f} MB) | AMP: Active | Batch: {self.batch_size}\n")

    def load_all_files(self):
        self.speech_files = list(self.speech_dir.rglob("*.flac")) + list(self.speech_dir.rglob("*.wav"))
        console.print(f"  Clean Speech Pool : {len(self.speech_files)} files")

        self.noise_files: Dict[str, List[Path]] = {}
        for cat_dir in sorted(self.noise_base.iterdir()):
            if cat_dir.is_dir():
                files = list(cat_dir.glob("*.wav")) + list(cat_dir.glob("*.flac")) + list(cat_dir.glob("*.mp3"))
                if files:
                    self.noise_files[cat_dir.name] = files
                    console.print(f"  Combat Noise [{cat_dir.name:18s}]: {len(files)} files")

        console.print()

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
        clean_speech = (clean_speech / peak_speech) * random.uniform(0.65, 0.85)

        # 2. Layer 1 to 4 combat noises
        num_noises = random.randint(1, min(4, len(self.noise_files)))
        active_cats = random.sample(list(self.noise_files.keys()), num_noises)

        noise_layers = []
        for cat in active_cats:
            n_file = random.choice(self.noise_files[cat])
            n_wav = self.load_audio_segment(n_file)
            n_peak = np.max(np.abs(n_wav)) + 1e-8
            gain = 10.0 ** (random.uniform(-8.0, 4.0) / 20.0)
            noise_layers.append((n_wav / n_peak) * 0.6 * gain)

        total_noise = sum(noise_layers) if noise_layers else np.zeros(samples, dtype=np.float32)

        if random.random() < 0.30:
            total_noise = np.tanh(total_noise * random.uniform(1.2, 1.8))

        mixture = clean_speech + total_noise

        mix_peak = np.max(np.abs(mixture))
        if mix_peak > 0.95:
            scale = 0.95 / mix_peak
            mixture *= scale
            clean_speech *= scale

        mix_stft = librosa.stft(mixture.astype(np.float32), n_fft=self.n_fft, hop_length=self.hop_length)
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
        console.print(f"[bold green]Starting {self.hours:.1f}-Hour High-Capacity GAN Training[/bold green]\n")

        start_time = time.time()
        target_seconds = self.hours * 3600.0

        while time.time() - start_time < target_seconds:
            batch_x, batch_y = self.generate_batch()
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # =========================================================
            # 1. TRAIN DISCRIMINATOR
            # =========================================================
            self.opt_d.zero_grad()
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                with torch.no_grad():
                    fake_y = self.generator(batch_x)

                # Real scores
                scores_real, _ = self.discriminator(batch_y)
                loss_d_real = sum(torch.mean((s - 1.0) ** 2) for s in scores_real) / len(scores_real)

                # Fake scores
                scores_fake, _ = self.discriminator(fake_y.detach())
                loss_d_fake = sum(torch.mean(s ** 2) for s in scores_fake) / len(scores_fake)

                loss_d = 0.5 * (loss_d_real + loss_d_fake)

            self.scaler.scale(loss_d).backward()
            self.scaler.unscale_(self.opt_d)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5.0)
            self.scaler.step(self.opt_d)

            # =========================================================
            # 2. TRAIN GENERATOR
            # =========================================================
            self.opt_g.zero_grad()
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                fake_y = self.generator(batch_x)

                # A. Content Reconstruction (Complex + Magnitude)
                pred_mag = torch.sqrt(fake_y[:, 0] ** 2 + fake_y[:, 1] ** 2 + 1e-8)
                tgt_mag = torch.sqrt(batch_y[:, 0] ** 2 + batch_y[:, 1] ** 2 + 1e-8)
                recon_loss = F.l1_loss(fake_y, batch_y)
                mag_loss = F.l1_loss(pred_mag, tgt_mag)
                hf_loss = F.l1_loss(pred_mag[:, 24:], tgt_mag[:, 24:])

                # B. Adversarial Loss (Fool the Discriminator)
                scores_gen, feats_fake = self.discriminator(fake_y)
                loss_adv = sum(torch.mean((s - 1.0) ** 2) for s in scores_gen) / len(scores_gen)

                # C. Feature Matching Loss (Texture Consistency)
                with torch.no_grad():
                    _, feats_real = self.discriminator(batch_y)

                loss_fm = 0.0
                for f_fake_scale, f_real_scale in zip(feats_fake, feats_real):
                    for f_f, f_r in zip(f_fake_scale, f_real_scale):
                        loss_fm += F.l1_loss(f_f, f_r)
                loss_fm = loss_fm / (len(feats_fake) * len(feats_fake[0]))

                # Total Generator Loss
                loss_g = (
                    0.30 * recon_loss +
                    0.30 * mag_loss +
                    0.20 * hf_loss +
                    0.10 * loss_adv +
                    1.00 * loss_fm
                )

            self.scaler.scale(loss_g).backward()
            self.scaler.unscale_(self.opt_g)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5.0)
            self.scaler.step(self.opt_g)

            self.scaler.update()
            self.sched_g.step()
            self.sched_d.step()

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
                    f"Adv: {loss_adv.item():.4f} | "
                    f"FM: {loss_fm.item():.4f} | "
                    f"Disc: {loss_d.item():.4f} | "
                    f"LR: {lr:.6f} | "
                    f"Elapsed: {elapsed/3600:.2f}h | "
                    f"Remaining: {remaining_h:.2f}h"
                )

            # Export audio & checkpoint every 1,000 steps
            if self.step % 1000 == 0:
                self.save_checkpoint(is_best=False)
                self.test_speech_extraction()
                self.clean_old_artifacts()

        self.save_checkpoint(is_best=True)
        self.test_speech_extraction()
        self.clean_old_artifacts()
        console.print(f"\n[bold green]✓ GAN Training Complete![/bold green] Total Steps: {self.step:,} | Best Loss: {self.best_loss:.5f}")

    def save_checkpoint(self, is_best: bool = False):
        filename = f"battlefield_gan_step{self.step}.pt" if not is_best else "battlefield_speech_specialist_best.pt"
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "step": self.step,
                "generator_state_dict": self.generator.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "loss": self.best_loss,
                "base_channels": self.base_channels,
                "domain": "battlefield_speech_gan",
            },
            str(path),
        )
        if not is_best:
            self.saved_checkpoints.append(path)
            console.print(f"  [green]✓ Saved Checkpoint:[/green] {path}")

    def test_speech_extraction(self):
        test_file = Path("examples/test_audio/long_combat_mission_benchmark.wav")
        if not test_file.exists():
            test_file = Path("examples/test_audio/podcast_battlefield_mix.wav")
        if not test_file.exists():
            return

        audio, sr = librosa.load(str(test_file), sr=self.sr, mono=True)
        chunk_samples = int(self.sr * self.segment_duration)

        self.generator.eval()
        with torch.no_grad():
            enhanced_chunks = []
            for i in range(0, len(audio), chunk_samples):
                chunk = audio[i : i + chunk_samples]
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

                stft = librosa.stft(chunk, n_fft=self.n_fft, hop_length=self.hop_length)
                features = np.stack([stft.real, stft.imag])
                input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                    pred = self.generator(input_tensor)

                pred_stft = pred[0, 0].float().cpu().numpy() + 1j * pred[0, 1].float().cpu().numpy()
                enhanced_chunk = librosa.istft(pred_stft, hop_length=self.hop_length)
                enhanced_chunks.append(enhanced_chunk)

            enhanced_audio = np.concatenate(enhanced_chunks)[: len(audio)]
            peak = np.max(np.abs(enhanced_audio))
            if peak > 0:
                enhanced_audio = enhanced_audio / peak * 0.9

            out_path = self.test_audio_dir / f"battlefield_speech_step{self.step}.wav"
            sf.write(str(out_path), enhanced_audio, self.sr)
            self.saved_test_wavs.append(out_path)
            console.print(f"  [green]✓ Exported GAN Test Audio:[/green] [bold]{out_path}[/bold]")

        self.generator.train()


if __name__ == "__main__":
    trainer = BattlefieldGANTrainer(hours=14.0, batch_size=4, base_channels=48, max_kept_checkpoints=2)
    trainer.train()
