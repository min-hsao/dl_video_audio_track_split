# Video Audio Track Split

A Python CLI tool to download videos from YouTube/other sites and separate audio tracks using Demucs AI. Perfect for extracting vocals, instrumentals, or individual stems.

## Features

- 📥 Download videos from YouTube and other platforms (yt-dlp)
- 🎵 AI-powered audio source separation (Demucs)
- 🎤 Extract vocals and accompaniment
- 🥁 Detailed mode: drums, bass, other, vocals
- 🎬 Optional video retention
- 📏 Resolution selection

## Requirements

- Python 3.8+
- FFmpeg
- CUDA (optional, for GPU acceleration)

## Installation

```bash
pip install -r requirements.txt

# Install FFmpeg (macOS)
brew install ffmpeg
```

## Usage

```bash
# Basic usage - extract vocals and accompaniment
python dlvats.py "https://youtube.com/watch?v=..."

# Keep original video file
python dlvats.py "https://youtube.com/watch?v=..." -kv

# Detailed mode - all 4 stems
python dlvats.py "https://youtube.com/watch?v=..." -d

# Custom output folder
python dlvats.py "https://youtube.com/watch?v=..." -of my_output

# Limit video resolution
python dlvats.py "https://youtube.com/watch?v=..." -r 720p
```

## Arguments

| Argument | Description |
|----------|-------------|
| `url` | Video URL to process |
| `-of, --output_folder` | Output folder (default: output) |
| `-d, --detailed` | Export all 4 stems |
| `-kv, --keep_video` | Save original video file |
| `-r, --resolution` | Max video resolution (e.g., 720p, 1080p) |

## Output

### Basic Mode
- `vocals.wav` - Isolated vocals
- `accompaniment.wav` - Everything except vocals

### Detailed Mode (-d)
- `vocals.wav` - Vocals
- `drums.wav` - Drums/percussion
- `bass.wav` - Bass
- `other.wav` - Other instruments

## Processing

Uses Demucs `htdemucs` model for high-quality separation:
- GPU acceleration when CUDA available
- Automatic resampling to model requirements
- Stereo output

## License

MIT License - see [LICENSE](LICENSE) for details.
