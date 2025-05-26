import os
import librosa
import numpy as np
import logging
import soundfile as sf
import warnings
import subprocess
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


def extract_audio_features(audio_path, sr=16000, n_fft=512, hop_length=256, n_mels=128):
    """
    Extract mel spectrogram and chroma features from audio file.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        n_mels: Number of mel bands
    
    Returns:
        tuple: (mel_spectrogram, chroma_features) as numpy arrays
    """
    try:
        # Load audio file
        try:
            y, sr_orig = sf.read(audio_path)
            if len(y.shape) > 1:  # Convert to mono if stereo
                y = y.mean(axis=1)
        except Exception as e:
            logger.warning("PySoundFile failed. Trying audioread instead.")
            y, sr_orig = librosa.load(audio_path, sr=sr, mono=True)
        
        # Resample if necessary
        if sr_orig != sr:
            y = librosa.resample(y=y, orig_sr=sr_orig, target_sr=sr)
        
        # Ensure audio length is sufficient for n_fft
        if len(y) < n_fft:
            warnings.warn(f"Audio length {len(y)} is shorter than n_fft {n_fft}. Padding with zeros.")
            y = np.pad(y, (0, n_fft - len(y)))
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=20,
            fmax=sr/2
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Compute chroma features
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Normalize chroma to [0, 1]
        chroma_norm = (chroma - chroma.min()) / (chroma.max() - chroma.min() + 1e-8)
        
        # Convert to float32
        mel_spec_norm = mel_spec_norm.astype(np.float32)
        chroma_norm = chroma_norm.astype(np.float32)
        
        logger.info(f"Audio file {audio_path} processed. Computed mel spectrogram (shape {mel_spec_norm.shape}) and chroma (shape {chroma_norm.shape}).")
        
        return mel_spec_norm, chroma_norm
        
    except FileNotFoundError:
        logger.error(f"Audio file not found: {audio_path}.")
        raise
    except Exception as e:
        logger.error(f"Error processing audio file {audio_path}: {str(e)}")
        raise

def extract_and_save_crow_audio(video_path, frame_time_seconds, fps, crow_id, frame_num, audio_dir, duration=2.0):
    """
    Extract audio segment from video when a crow is detected.
    
    Args:
        video_path: Path to the video file
        frame_time_seconds: Time in video when crow was detected (seconds)
        fps: Frames per second of the video
        crow_id: ID of the detected crow
        frame_num: Frame number where crow was detected
        audio_dir: Directory to save audio files
        duration: Duration of audio segment to extract (seconds)
    
    Returns:
        str: Path to saved audio file, or None if extraction failed
    """
    try:
        if not video_path or not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return None
        
        # Create audio directory for this crow
        crow_audio_dir = Path(audio_dir) / crow_id
        crow_audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate start time (center the detection)
        start_time = max(0, frame_time_seconds - duration/2)
        
        # Generate output filename
        audio_filename = f"frame_{frame_num:06d}_{start_time:.2f}s.wav"
        output_path = crow_audio_dir / audio_filename
        
        # Use ffmpeg to extract audio segment
        try:
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-ss', str(start_time),
                '-t', str(duration),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file
                str(output_path)
            ]
            
            # Run ffmpeg with suppressed output
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and output_path.exists():
                logger.debug(f"Extracted audio segment: {output_path}")
                return str(output_path)
            else:
                logger.warning(f"ffmpeg failed to extract audio: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.warning(f"Audio extraction timed out for {video_path}")
            return None
        except FileNotFoundError:
            logger.warning("ffmpeg not found. Audio extraction disabled.")
            return None
        except Exception as e:
            logger.warning(f"Error running ffmpeg: {e}")
            return None
            
    except Exception as e:
        logger.error(f"Error extracting audio from {video_path}: {e}")
        return None
