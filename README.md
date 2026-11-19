# Phonos — Ultra-Compact Universal 44.1kHz Speech Restorer

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT" height="20"></a>
  <a href="https://huggingface.co/jaadu611/phonos"><img src="https://img.shields.io/badge/Model%20Size-78.7%20MB-success.svg" alt="Model Size" height="20"></a>
  <a href="https://huggingface.co/jaadu611/phonos"><img src="https://img.shields.io/badge/Model-phonos__v1.0.pt-2ea44f.svg" alt="Model Weights" height="20"></a>
  <a href="#universal-restoration-capabilities"><img src="https://img.shields.io/badge/GPU%20Speed-11.8x%20Real--Time-brightgreen.svg" alt="GPU Speed" height="20"></a>
  <a href="#universal-restoration-capabilities"><img src="https://img.shields.io/badge/CPU%20Speed-1.1x%20Real--Time-blue.svg" alt="CPU Speed" height="20"></a>
  <a href="#universal-restoration-capabilities"><img src="https://img.shields.io/badge/Fidelity-44.1%20kHz%20Studio-orange.svg" alt="Audio Quality" height="20"></a>
</p>

**Phonos** is an ultra-compact, quantized distribution of the state-of-the-art **[VoiceFixer](https://github.com/haoheliu/voicefixer)** 44.1kHz universal speech restoration model. Unlike standard subtractive noise filters that only attenuate background hiss, Phonos actively reconstructs damaged or missing speech harmonics up to full 44.1kHz studio bandwidth.

Operating on a selective **Mixed NF4 / FP8 Precision Architecture**, Phonos delivers the complete studio-grade speech recovery of VoiceFixer in a compact **78.7 MB package** (compressed from ~440 MB). It runs at **11.8x real-time (<950 MB VRAM)** on consumer GPUs, and **1.1x real-time (<450 MB RAM, 0 MB VRAM)** on pure CPU — making it lightweight enough for mobile edge hardware, standard laptops without dedicated GPUs, and low-resource environments.

---

## Universal Restoration Capabilities

Phonos addresses the four core degradation modes of real-world audio:

| Degradation Challenge | Problem in Raw Recording | Phonos Restoration Mechanism |
|:---|:---|:---|
| **Bandwidth Truncation** | Narrowband 3.4kHz–8kHz phone/radio audio sounds muffled | **44.1 kHz Resynthesis**: Generates missing high-frequency harmonics (0–22.05 kHz) using learned neural vocoder priors. |
| **Acoustic Reverberation** | Echo, room reflections, and cavernous hall acoustics | **Dereverberation**: Deconvolves long impulse responses ($T_{60} > 0.8\text{s}$) to restore upfront direct-path studio proximity. |
| **Waveform Clipping** | Harsh distortion and flat-topped peak saturation | **Peak Inpainting**: Reconstructs rounded wave crests and strips harmonic splatter. |
| **Extreme Background Noise** | Babble, machinery, and non-stationary noise at 0 dB SNR | **Neural Denoising**: Attenuates interference without phase degradation. |

---

## Audio Demonstration Suite (`examples/`)

Verified across authentic degradation scenarios in `examples/`:

| Scenario (`examples/`) | Input Degradation | Restored Output (44.1kHz Studio) |
|:---|:---|:---|
| **01_telephone_bandwidth_extension** | Muffled 3.4kHz telephone lowpass filter | Full-bandwidth 44.1kHz vocal presence with high-shelf air |
| **02_cavernous_room_dereverberation** | Large hall echo ($T_{60} = 0.85\text{s}$) | Dry, intimate vocal isolation with room echo stripped |
| **03_severe_clipped_distortion** | Hard-clipped waveform (-14 dB saturation) | Restored natural dynamics and harmonic smoothness |
| **04_extreme_ambient_noise** | Equal-volume non-stationary babble (0 dB SNR) | Isolated voice with background noise floor eliminated |

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/jaadu611/phonos
cd phonos

# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Checkpoint (`phonos_v1.0.pt`)

The model checkpoint (`phonos_v1.0.pt`, 78.7 MB) is automatically fetched on first run from [Hugging Face](https://huggingface.co/jaadu611/phonos) or GitHub Releases.

Optional manual download:
```bash
mkdir -p checkpoints
wget -O checkpoints/phonos_v1.0.pt https://huggingface.co/jaadu611/phonos/resolve/main/phonos_v1.0.pt
```

### 3. Usage

#### Restore Any Recording to 44.1kHz Studio Audio
```bash
# Standard studio enhancement (with harmonic presence exciter)
python cli.py recording.wav -o restored_studio.wav
# or
python cli.py enhance recording.wav -o restored_studio.wav

# Disable harmonic exciter (raw neural restoration)
python cli.py enhance recording.wav --no-exciter -o restored_studio.wav
```

#### Inspect Audio File Metadata
```bash
python cli.py info recording.wav
```

---

## Architecture & Performance Specifications

* **Quantization Scheme**: Selective Mixed 4-bit NormalFloat (NF4) for bottleneck stages; 8-bit Float (FP8) for outer stages and neural vocoder.
* **GPU Throughput**: 11.8x real-time on NVIDIA RTX 3050 Laptop GPU (<950 MB VRAM).
* **CPU Throughput**: 1.1x real-time on standard laptop CPU (0 MB VRAM, <450 MB RAM).
* **Normalization Statistics**: 16-bit Float (FP16).
* **Chunking Runtime**: 5.0-second window with automatic reflection padding for short remainders.

---

## Acknowledgments & Attribution

Phonos is built directly on the foundational speech restoration research and model architecture of **[VoiceFixer](https://github.com/haoheliu/voicefixer)**. All core neural modeling, training, vocoder synthesis, and baseline weights are the work of Haohe Liu and collaborators:

> **VoiceFixer: A Unified Framework for High-Fidelity Speech Restoration**  
> *Haohe Liu, Xubo Liu, Qiuqiang Kong, Qiao Tian, Yan Zhao, DeLiang Wang, Chuanzeng Huang, Yuxuan Wang*  
> Interspeech 2022 — [Paper on arXiv](https://arxiv.org/abs/2109.13731) | [Official Repository](https://github.com/haoheliu/voicefixer)

Phonos contributes post-training optimizations on top of VoiceFixer:
* **Selective Mixed NF4/FP8 Quantization**: Compresses the original ~440 MB checkpoint down to 78.7 MB.
* **Pruned Low-Memory Inference**: Slices GPU memory footprint to under 950 MB VRAM for consumer laptop GPUs.
* **Harmonic Presence Exciter**: Post-processing saturation for high-frequency clarity.
* **Standalone CLI**: Streamlined, single-file interface without heavy framework overhead.

---

## License

This project is licensed under the [MIT License](LICENSE).
