# AudioForge

Forensic audio separation and analysis toolkit. Extract, isolate, and amplify specific sounds from complex audio mixtures using text queries.

## Features

- **Analyze**: Detect what sounds are present in any audio
- **Extract**: Isolate specific sounds using natural language queries
- **Amplify**: Boost extracted sounds for analysis
- **Remove**: Strip unwanted sounds from recordings

## Installation

```bash
git clone https://github.com/yourusername/audioforge
cd audioforge
pip install -e .
```

## Usage

```bash
# Analyze what's in an audio file
audioforge analyze recording.wav

# Extract helicopter sounds
audioforge extract recording.wav --query "helicopter rotor sound"

# Extract and amplify gunfire
audioforge extract recording.wav --query "gunshots" --gain 20

# Remove speech, keep background
audioforge extract recording.wav --query "all sounds except human speech"
```

## Hardware Requirements

- 4GB+ VRAM (tested on RTX 3050)
- 16GB RAM
- CPU-only mode available (slower)

## Architecture

```
Input Audio → Analyzer (detect sounds) → Separator (extract) → Amplifier → Output
```

## License

MIT
