"""
AudioForge Battlefield Specialist Training Engine.
Features live directory scanning, automatic dataset manifest generation, and self-cleaning storage.
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

from specialists.base_unet import SmallUNet

console = Console()


class BattlefieldSpecialistTrainer:
    """
    Dynamically scans training_data/battlefield/ and trains a self-cleaning universal specialist.
    """

    def __init__(
        self,
        hours: float = 14.0,
        max_kept_checkpoints: int = 2,
        speech_dir: str = "training_data/clean_speech",
        noise_base_dir: str = "training_data/battlefield",
    ):
        self.hours = hours
        self.max_kept_checkpoints = max_kept_checkpoints
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.speech_dir = Path(speech_dir)
        self.noise_base = Path(noise_base_dir)

        self.specialist_dir = Path("specialists/battlefield")
        self.checkpoint_dir = self.specialist_dir / "checkpoints"
        self.test_audio_dir = Path("examples/test_audio")

        self.specialist_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.test_audio_dir.mkdir(parents=True, exist_ok=True)

        self.saved_checkpoints: List[Path] = []
        self.saved_test_wavs: List[Path] = []

        # Audio settings
        self.sr = 16000
        self.segment_duration = 2.0
        self.n_fft = 512
        self.hop_length = 160
        self.batch_size = 4

        # Dynamic stem indexing
        self.files_by_stem: Dict[int, List[Path]] = {}
        self.stem_map: Dict[str, int] = {}
        self.stem_names: List[str] = []

        # Scan directories and write initial manifest
        self.scan_and_update_manifest()

        # Model (~2.1M parameters)
        self.model = SmallUNet(
            base_channels=16,
            num_classes=max(16, len(self.stem_names)),
            mode="generate",
        ).to(self.device)
        self.model.set_stem_map(self.stem_map)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            weight_decay=1e-5,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=120000,
            eta_min=1e-5,
        )

        self.step = 0
        self.best_loss = float("inf")

        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        console.print(f"[bold blue]AudioForge Battlefield Specialist Trainer[/bold blue]")
        console.print(f"Model: SmallUNet (Dynamic FiLM) | Params: {params:,} (~{params*4/1024/1024:.2f} MB)")
        console.print(f"Device: {self.device} | Target: {self.hours:.1f} hours | Manifest: {self.specialist_dir / 'manifest.json'}\n")

    def scan_and_update_manifest(self):
        """Scans directories live, builds category index, and saves manifest.json"""
        self.files_by_stem.clear()
        self.stem_map.clear()
        self.stem_names.clear()

        manifest_data = {
            "domain": "battlefield",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "categories": {},
        }

        # 0: Clean Speech
        speech_files = list(self.speech_dir.rglob("*.flac")) + list(self.speech_dir.rglob("*.wav"))
        self.files_by_stem[0] = speech_files
        self.stem_map["speech"] = 0
        self.stem_names.append("speech")
        manifest_data["categories"]["speech"] = {
            "stem_id": 0,
            "file_count": len(speech_files),
            "sample_files": [f.name for f in speech_files[:5]],
        }

        # 1+: Dynamically discovered noise folders
        cid = 1
        for cat_dir in sorted(self.noise_base.iterdir()):
            if cat_dir.is_dir():
                cat_name = cat_dir.name
                files = list(cat_dir.glob("*.wav")) + list(cat_dir.glob("*.flac")) + list(cat_dir.glob("*.mp3")) + list(cat_dir.glob("*.ogg"))
                if files:
                    self.files_by_stem[cid] = files
                    self.stem_map[cat_name] = cid
                    self.stem_names.append(cat_name)
                    manifest_data["categories"][cat_name] = {
                        "stem_id": cid,
                        "file_count": len(files),
                        "sample_files": [f.name for f in files[:5]],
                    }
                    cid += 1

        manifest_data["total_categories"] = len(self.stem_names)
        manifest_data["total_files"] = sum(len(f) for f in self.files_by_stem.values())

        # Write manifest.json
        manifest_path = self.specialist_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2)

        console.print(f"[bold green]Dataset Manifest Updated:[/bold green] {len(self.stem_names)} categories, {manifest_data['total_files']} total audio files.")
        for name in self.stem_names:
            console.print(f"  [{self.stem_map[name]:2d}] {name:22s}: {len(self.files_by_stem[self.stem_map[name]])} files")
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

    def generate_pair(self) -> Tuple[np.ndarray, np.ndarray, int]:
        samples = int(self.sr * self.segment_duration)
        available_ids = list(self.files_by_stem.keys())

        # Pick 2 to 5 random sounds to mix
        num_sounds = random.randint(2, min(5, len(available_ids)))
        active_ids = random.sample(available_ids, num_sounds)

        # Select extraction target stem
        target_id = random.choice(active_ids)

        stem_waveforms = {}
        for s_id in active_ids:
            files = self.files_by_stem.get(s_id, [])
            if files:
                wav = self.load_audio_segment(random.choice(files))
            else:
                wav = np.random.randn(samples) * 0.05

            gain = 10.0 ** (random.uniform(-6.0, 6.0) / 20.0)
            peak = np.max(np.abs(wav)) + 1e-8
            stem_waveforms[s_id] = (wav / peak) * 0.7 * gain

        target_waveform = stem_waveforms[target_id]
        mixture = sum(stem_waveforms.values())

        mix_peak = np.max(np.abs(mixture))
        if mix_peak > 0.95:
            scale = 0.95 / mix_peak
            mixture *= scale
            target_waveform *= scale

        mix_stft = librosa.stft(mixture.astype(np.float32), n_fft=self.n_fft, hop_length=self.hop_length)
        tgt_stft = librosa.stft(target_waveform.astype(np.float32), n_fft=self.n_fft, hop_length=self.hop_length)

        return (
            np.stack([mix_stft.real, mix_stft.imag], axis=0),
            np.stack([tgt_stft.real, tgt_stft.imag], axis=0),
            target_id,
        )

    def generate_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_x, batch_y, batch_prompts = [], [], []
        for _ in range(self.batch_size):
            x, y, prompt_id = self.generate_pair()
            batch_x.append(x)
            batch_y.append(y)
            batch_prompts.append(prompt_id)

        return (
            torch.tensor(np.array(batch_x), dtype=torch.float32),
            torch.tensor(np.array(batch_y), dtype=torch.float32),
            torch.tensor(batch_prompts, dtype=torch.long),
        )

    def compute_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        pred_mag = torch.sqrt(predicted[:, 0] ** 2 + predicted[:, 1] ** 2 + 1e-8)
        tgt_mag = torch.sqrt(target[:, 0] ** 2 + target[:, 1] ** 2 + 1e-8)

        recon_loss = F.l1_loss(predicted, target)
        mag_loss = F.l1_loss(pred_mag, tgt_mag)

        inactive_mask = tgt_mag < 0.01
        if inactive_mask.any():
            crosstalk_loss = pred_mag[inactive_mask].mean() * 4.0
        else:
            crosstalk_loss = torch.tensor(0.0, device=self.device)

        total_loss = 0.40 * recon_loss + 0.40 * mag_loss + 0.20 * crosstalk_loss

        return total_loss, {
            "recon": recon_loss.item(),
            "mag": mag_loss.item(),
            "crosstalk": crosstalk_loss.item(),
        }

    def clean_old_artifacts(self):
        """Self-cleaning storage management"""
        while len(self.saved_checkpoints) > self.max_kept_checkpoints:
            old_ckpt = self.saved_checkpoints.pop(0)
            if old_ckpt.exists() and "best" not in old_ckpt.name:
                try:
                    old_ckpt.unlink()
                except Exception:
                    pass

        while len(self.saved_test_wavs) > 6:
            old_wav = self.saved_test_wavs.pop(0)
            if old_wav.exists():
                try:
                    old_wav.unlink()
                except Exception:
                    pass

    def train(self):
        console.print(f"[bold green]Starting {self.hours:.1f}-Hour Battlefield Specialist Training[/bold green]\n")

        start_time = time.time()
        target_seconds = self.hours * 3600.0

        while time.time() - start_time < target_seconds:
            # Rescan directory pool periodically (every 5000 steps)
            if self.step % 5000 == 0 and self.step > 0:
                self.scan_and_update_manifest()
                self.model.set_stem_map(self.stem_map)

            batch_x, batch_y, batch_prompts = self.generate_batch()
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_prompts = batch_prompts.to(self.device)

            self.optimizer.zero_grad()
            predicted = self.model(batch_x, prompt=batch_prompts)

            loss, metrics = self.compute_loss(predicted, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

            self.optimizer.step()
            self.scheduler.step()

            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.save_checkpoint(is_best=True)

            elapsed = time.time() - start_time
            self.step += 1

            if self.step % 200 == 0 or self.step == 1:
                remaining_h = max(0.0, (target_seconds - elapsed) / 3600.0)
                lr = self.scheduler.get_last_lr()[0]
                console.print(
                    f"  Step [bold cyan]{self.step:6d}[/bold cyan] | "
                    f"Loss: [bold yellow]{loss.item():.5f}[/bold yellow] | "
                    f"Best: [bold green]{self.best_loss:.5f}[/bold green] | "
                    f"Recon: {metrics['recon']:.4f} | "
                    f"LR: {lr:.6f} | "
                    f"Elapsed: {elapsed/3600:.2f}h | "
                    f"Remaining: {remaining_h:.2f}h"
                )

            # Checkpoint, test audio & cleanup every 2,000 steps
            if self.step % 2000 == 0:
                self.save_checkpoint(is_best=False)
                self.test_multi_stems()
                self.clean_old_artifacts()

        self.save_checkpoint(is_best=True)
        self.test_multi_stems()
        self.clean_old_artifacts()
        console.print(f"\n[bold green]✓ Battlefield Training Complete![/bold green] Total Steps: {self.step:,} | Best Loss: {self.best_loss:.5f}")

    def save_checkpoint(self, is_best: bool = False):
        filename = f"battlefield_step{self.step}.pt" if not is_best else "battlefield_specialist_best.pt"
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "step": self.step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.best_loss,
                "domain": "battlefield",
                "stem_map": self.stem_map,
                "stem_names": self.stem_names,
            },
            str(path),
        )
        if not is_best:
            self.saved_checkpoints.append(path)
            console.print(f"  [green]✓ Saved Checkpoint:[/green] {path}")

    def test_multi_stems(self):
        """Extract Speech, Gunfire, and Explosions from composite mix"""
        test_file = Path("examples/test_audio/podcast_battlefield_mix.wav")
        if not test_file.exists():
            return

        audio, sr = librosa.load(str(test_file), sr=self.sr, mono=True)
        chunk_samples = int(self.sr * self.segment_duration)

        self.model.eval()
        with torch.no_grad():
            for target_stem in ["speech", "gunfire", "explosions"]:
                enhanced_chunks = []
                for i in range(0, len(audio), chunk_samples):
                    chunk = audio[i : i + chunk_samples]
                    if len(chunk) < chunk_samples:
                        chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

                    stft = librosa.stft(chunk, n_fft=self.n_fft, hop_length=self.hop_length)
                    features = np.stack([stft.real, stft.imag])
                    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

                    pred = self.model(input_tensor, prompt=target_stem)
                    pred_stft = pred[0, 0].cpu().numpy() + 1j * pred[0, 1].cpu().numpy()
                    enhanced_chunk = librosa.istft(pred_stft, hop_length=self.hop_length)
                    enhanced_chunks.append(enhanced_chunk)

                enhanced_audio = np.concatenate(enhanced_chunks)[: len(audio)]
                peak = np.max(np.abs(enhanced_audio))
                if peak > 0:
                    enhanced_audio = enhanced_audio / peak * 0.9

                out_path = self.test_audio_dir / f"battlefield_step{self.step}_{target_stem}.wav"
                sf.write(str(out_path), enhanced_audio, self.sr)
                self.saved_test_wavs.append(out_path)
                console.print(f"  [green]✓ Exported {target_stem.upper()} Stem:[/green] [bold]{out_path}[/bold]")

        self.model.train()


if __name__ == "__main__":
    trainer = BattlefieldSpecialistTrainer(hours=14.0, max_kept_checkpoints=2)
    trainer.train()
