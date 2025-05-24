import os
import librosa
import numpy as np
import logging
import soundfile as sf
import warnings
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_audio_segment_from_video(video_path, start_time, duration, output_path=None, sr=16000):
    """
    Extract audio segment from video file at specific timestamp.
    
    Args:
        video_path: Path to video file
        start_time: Start time in seconds
        duration: Duration of audio segment in seconds
        output_path: Output path for audio file (optional, will create temp file if None)
        sr: Sample rate for output audio
    
    Returns:
        tuple: (audio_array, sample_rate, output_path) or None if extraction fails
    """
    try:
        # Create temporary output file if none provided
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        # Use ffmpeg to extract audio segment
        cmd = [
            'ffmpeg', '-y',  # Overwrite output file
            '-i', str(video_path),
            '-ss', str(start_time),  # Start time
            '-t', str(duration),     # Duration
            '-vn',                   # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit little endian
            '-ar', str(sr),          # Sample rate
            '-ac', '1',              # Mono
            str(output_path)
        ]
        
        # Run ffmpeg with error handling
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"FFmpeg extraction failed: {result.stderr}")
            return None
        
        # Verify the output file exists and has content
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            logger.warning(f"Audio extraction produced empty file: {output_path}")
            return None
        
        # Load the extracted audio
        try:
            y, sr_actual = sf.read(output_path)
            if len(y.shape) > 1:  # Convert to mono if stereo
                y = y.mean(axis=1)
        except Exception as e:
            logger.warning(f"Could not read extracted audio with soundfile: {e}")
            try:
                y, sr_actual = librosa.load(output_path, sr=sr, mono=True)
            except Exception as e2:
                logger.error(f"Could not read extracted audio with librosa: {e2}")
                return None
        
        logger.debug(f"Extracted audio segment: {len(y)} samples at {sr_actual} Hz")
        return y, sr_actual, output_path
        
    except Exception as e:
        logger.error(f"Error extracting audio segment from {video_path}: {str(e)}")
        return None


def extract_and_save_crow_audio(video_path, frame_time, fps, crow_id, frame_num, audio_dir, duration=2.0):
    """
    Extract audio segment corresponding to a crow crop and save it.
    
    Args:
        video_path: Path to source video file
        frame_time: Time of the frame in seconds (or datetime object)
        fps: Video frame rate
        crow_id: ID of the crow
        frame_num: Frame number
        audio_dir: Directory to save audio files
        duration: Duration of audio segment to extract (seconds)
    
    Returns:
        str: Path to saved audio file, or None if extraction failed
    """
    try:
        # Convert frame_time to seconds if it's not already
        if hasattr(frame_time, 'timestamp'):  # datetime object
            # For datetime objects, we need to calculate relative time
            # This is a fallback - ideally frame_time should be in seconds
            frame_time_seconds = frame_num / fps if fps > 0 else 0
        elif isinstance(frame_time, (int, float)):
            frame_time_seconds = float(frame_time)
        else:
            # Default to calculating from frame number
            frame_time_seconds = frame_num / fps if fps > 0 else 0
        
        # Create audio directory for this crow
        audio_dir = Path(audio_dir)
        crow_audio_dir = audio_dir / crow_id
        crow_audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        audio_filename = f"frame_{frame_num:06d}.wav"
        audio_output_path = crow_audio_dir / audio_filename
        
        # Calculate start time (center the audio around the frame time)
        start_time = max(0, frame_time_seconds - duration / 2)
        
        # Extract audio segment
        result = extract_audio_segment_from_video(
            video_path, start_time, duration, str(audio_output_path)
        )
        
        if result is None:
            logger.warning(f"Failed to extract audio for crow {crow_id} at frame {frame_num}")
            return None
        
        _, _, output_path = result
        logger.debug(f"Saved audio segment for crow {crow_id}: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error extracting crow audio: {str(e)}")
        return None


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
