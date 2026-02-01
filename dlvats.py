#Dlvats.py
#download video audio track split
#by Min-Hsao Chen (w/ deepseek R1)
#v0.005 (01/31/2025)

import argparse
import torch
import yt_dlp
import os
import time
import threading
import subprocess
from datetime import datetime
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio
import warnings
warnings.filterwarnings("ignore")

# Global flag for spinner animation
stop_event = threading.Event()

def download_spinner():
    """Show animated dots while processing"""
    while not stop_event.is_set():
        for i in range(3):
            print('.', end='', flush=True)
            time.sleep(0.5)
        print('\b\b\b   \b\b\b', end='', flush=True)

def check_codec_compatibility(filename):
    """Verify video codec and fix HEVC tag if needed"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=codec_name,codec_tag_string',
             '-of', 'csv=p=0', filename],
            capture_output=True, text=True
        )
        output = result.stdout.strip()
        if ',' not in output:
            print(f"⚠️  No codec information found for {filename}")
            return
        codec, tag = output.split(',', 1)
        if codec == 'hevc' and tag != 'hvc1':
            print("⚠️  Fixing HEVC tag for QuickTime compatibility...")
            base, ext = os.path.splitext(filename)
            new_filename = f"{base}_quicktime{ext}"
            subprocess.run([
                'ffmpeg', '-i', filename,
                '-c', 'copy', '-tag:v', 'hvc1',
                new_filename
            ], check=True)
            os.replace(new_filename, filename)
            print("✅  HEVC tag fixed")
    except Exception as e:
        print(f"⚠️  Codec check failed: {e}")

def download_video(url, output_dir, keep_video=False, resolution=None):
    """Download video/audio and return path to audio file"""

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define paths
    audio_path = os.path.join(output_dir, 'temp_audio.wav')
    video_path = os.path.join(output_dir, 'output.mp4')

    try:
        if keep_video:
            # Download video first
            video_opts = {
                'quiet': True,
                'outtmpl': video_path,
                'merge_output_format': 'mp4',
                'postprocessor_args': ['-movflags', '+faststart'],
            }

            # Set format based on resolution
            if resolution:
                res = ''.join(filter(str.isdigit, resolution))
                if res:
                    video_opts['format'] = f'bestvideo[ext=mp4][vcodec^=avc][height<={res}]+bestaudio[ext=m4a]/best[ext=mp4]'
            else:
                video_opts['format'] = 'bestvideo[ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/best[ext=mp4]'

            # Download video
            with yt_dlp.YoutubeDL(video_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'video')
                print(f"⏬ Downloading: {title}...")
                ydl.download([url])
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video file not found: {video_path}")
                check_codec_compatibility(video_path)
                print(f"✅ Download complete: {os.path.basename(video_path)}")
                print(f"📏 File size: {os.path.getsize(video_path)//1024}KB")

        # Download/extract audio
        audio_opts = {
            'quiet': True,
            'outtmpl': os.path.join(output_dir, 'temp_audio.%(ext)s'),
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }]
        }

        with yt_dlp.YoutubeDL(audio_opts) as ydl:
            ydl.download([url])

        return audio_path

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

def separate_audio(audio_path, output_dir, detailed=False):
    global stop_event
    start_time = time.time()

    # Start spinner thread
    stop_event.clear()
    spinner = threading.Thread(target=download_spinner)
    spinner.daemon = True
    spinner.start()

    try:
        # Load model and set device
        model = get_model('htdemucs')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)

        # Ensure stereo input (2 channels)
        if waveform.shape[0] == 1:  # Convert mono to stereo
            waveform = torch.cat([waveform, waveform], dim=0)
        elif waveform.shape[0] > 2:  # Handle multi-channel audio
            waveform = waveform[:2, :]

        # Resample if needed
        if sample_rate != model.samplerate:
            resampler = torchaudio.transforms.Resample(sample_rate, model.samplerate)
            waveform = resampler(waveform)

        # Prepare tensor for model
        waveform = waveform.unsqueeze(0).to(device)

        # Separate tracks
        with torch.no_grad():
            sources = apply_model(model, waveform)
            sources = sources.cpu()

        # Process output stems
        sources = sources.squeeze(0)

        # Save results
        if detailed:
            source_names = ["drums", "bass", "other", "vocals"]
            for idx, name in enumerate(source_names):
                stem = sources[idx]
                output_path = os.path.join(output_dir, f"{name}.wav")
                torchaudio.save(output_path, stem, model.samplerate)
        else:
            vocals = sources[3]
            accompaniment = sources[:3].sum(dim=0)
            torchaudio.save(os.path.join(output_dir, "vocals.wav"), vocals, model.samplerate)
            torchaudio.save(os.path.join(output_dir, "accompaniment.wav"), accompaniment, model.samplerate)

        # Calculate processing time
        processing_time = time.time() - start_time
        mins, secs = divmod(processing_time, 60)
        time_str = f"{int(mins)}m {int(secs)}s"

        return time_str

    finally:
        stop_event.set()
        spinner.join()

def main():
    parser = argparse.ArgumentParser(
        description='Video downloader and audio separator with Demucs AI',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog='''Examples:
  Basic usage:  python script.py "https://youtube.com/..."
  Full options: python script.py "https://youtube.com/..." -of results -kv -d -r 1080p'''
    )
    parser.add_argument('url', help='URL of the video to process')
    parser.add_argument('-of', '--output_folder', default='output',
                      help='Output folder (default: output)')
    parser.add_argument('-d', '--detailed', action='store_true',
                      help='Export all 4 stems (drums, bass, other, vocals)')
    parser.add_argument('-kv', '--keep_video', action='store_true',
                      help='Save original video file along with audio')
    parser.add_argument('-r', '--resolution',
                      help='Max video resolution (e.g., 720p, 1080p)')
    args = parser.parse_args()

    try:
        print(f"\n{' Starting Process ':-^50}")
        print("Downloading content...", end='', flush=True)

        # Create output directory
        os.makedirs(args.output_folder, exist_ok=True)

        # Download and process
        actual_audio_path = download_video(args.url, args.output_folder, args.keep_video, args.resolution)

        print(f"\n\n{' Audio Separation ':-^50}")
        print(f"Processing device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        print("Separating tracks", end='')

        time_taken = separate_audio(actual_audio_path, args.output_folder, args.detailed)

        print(f"\n\n{' Completion Stats ':-^50}")
        print(f"Total processing time: {time_taken}")
        print(f"Output folder: {os.path.abspath(args.output_folder)}")
        print(f"{' Process Complete ':-^50}")

    except Exception as e:
        print(f"\n\n{' ERROR ':-^50}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup temp audio file
        if 'actual_audio_path' in locals() and os.path.exists(actual_audio_path):
            os.remove(actual_audio_path)

if __name__ == "__main__":
    main()
