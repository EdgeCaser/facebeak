#!/usr/bin/env python3
"""
Standalone Audio Filter for Crow Training Data

This script analyzes audio files extracted from crow videos and filters out
obvious non-crow sounds like traffic, human speech, music, etc.

Usage:
    python audio_filter.py /path/to/audio/directory
    python audio_filter.py /path/to/audio/directory --output-dir filtered_audio
    python audio_filter.py /path/to/audio/directory --action move --threshold 0.7
"""

import os
import sys
import argparse
import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from datetime import datetime
import json
import shutil
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AudioClassifier:
    """Classify audio segments as crow vs non-crow sounds."""
    
    def __init__(self):
        self.sr = 16000
        self.n_fft = 512
        self.hop_length = 256
        self.n_mels = 128
        
    def extract_features(self, audio_path):
        """Extract audio features for classification."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            
            if len(y) < self.n_fft:
                logger.warning(f"Audio too short: {audio_path}")
                return None
                
            features = {}
            
            # 1. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            
            # 2. MFCC features (good for distinguishing speech vs bird sounds)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            
            # 3. Zero crossing rate (helps distinguish periodic vs aperiodic sounds)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # 4. Chroma features (musical content detection)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma)
            features['chroma_std'] = np.std(chroma)
            
            # 5. Spectral contrast (texture)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['contrast_mean'] = np.mean(contrast)
            
            # 6. Tonnetz (harmonic content)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            features['tonnetz_mean'] = np.mean(tonnetz)
            
            # 7. Energy and power
            features['rms_energy'] = np.mean(librosa.feature.rms(y=y)[0])
            features['total_energy'] = np.sum(y**2)
            
            # 8. Frequency domain analysis
            fft = np.fft.fft(y)
            magnitude = np.abs(fft)
            frequencies = np.fft.fftfreq(len(fft), 1/sr)
            
            # Dominant frequency
            dominant_freq_idx = np.argmax(magnitude[:len(magnitude)//2])
            features['dominant_frequency'] = frequencies[dominant_freq_idx]
            
            # Energy in different frequency bands
            low_freq_energy = np.sum(magnitude[(frequencies >= 0) & (frequencies < 500)])
            mid_freq_energy = np.sum(magnitude[(frequencies >= 500) & (frequencies < 2000)])
            high_freq_energy = np.sum(magnitude[(frequencies >= 2000) & (frequencies < 8000)])
            
            total_freq_energy = low_freq_energy + mid_freq_energy + high_freq_energy
            if total_freq_energy > 0:
                features['low_freq_ratio'] = low_freq_energy / total_freq_energy
                features['mid_freq_ratio'] = mid_freq_energy / total_freq_energy
                features['high_freq_ratio'] = high_freq_energy / total_freq_energy
            else:
                features['low_freq_ratio'] = 0
                features['mid_freq_ratio'] = 0
                features['high_freq_ratio'] = 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {e}")
            return None
    
    def classify_audio(self, features):
        """
        Classify audio as crow vs non-crow based on heuristics.
        Returns confidence score (0-1, higher = more likely to be crow)
        """
        if features is None:
            return 0.0
        
        score = 0.5  # Start neutral
        confidence_factors = []
        
        # 1. Frequency analysis
        # Crows typically vocalize in 1-4kHz range
        dominant_freq = features.get('dominant_frequency', 0)
        if 800 <= dominant_freq <= 4000:
            score += 0.15
            confidence_factors.append(f"Good dominant freq: {dominant_freq:.0f}Hz")
        elif dominant_freq < 200 or dominant_freq > 6000:
            score -= 0.2
            confidence_factors.append(f"Poor dominant freq: {dominant_freq:.0f}Hz")
        
        # 2. Spectral characteristics
        # Birds tend to have higher spectral centroids than speech or traffic
        spectral_centroid = features.get('spectral_centroid_mean', 0)
        if spectral_centroid > 2000:
            score += 0.1
            confidence_factors.append("High spectral centroid (bird-like)")
        elif spectral_centroid < 1000:
            score -= 0.15
            confidence_factors.append("Low spectral centroid (speech/traffic-like)")
        
        # 3. MFCC analysis for speech detection
        # Human speech has characteristic MFCC patterns
        mfcc_1_mean = features.get('mfcc_1_mean', 0)
        mfcc_2_mean = features.get('mfcc_2_mean', 0)
        
        # Speech typically has specific MFCC patterns
        if abs(mfcc_1_mean) > 50 or abs(mfcc_2_mean) > 30:
            score -= 0.2
            confidence_factors.append("Speech-like MFCC pattern")
        
        # 4. Harmonic content (music detection)
        chroma_mean = features.get('chroma_mean', 0)
        tonnetz_mean = features.get('tonnetz_mean', 0)
        
        # High harmonic content suggests music
        if chroma_mean > 0.3 and tonnetz_mean > 0.2:
            score -= 0.25
            confidence_factors.append("High harmonic content (music-like)")
        
        # 5. Zero crossing rate
        # Birds have more varied zero crossing patterns than steady sounds
        zcr_std = features.get('zcr_std', 0)
        if zcr_std > 0.01:
            score += 0.1
            confidence_factors.append("Variable zero crossing (bird-like)")
        elif zcr_std < 0.005:
            score -= 0.1
            confidence_factors.append("Steady zero crossing (mechanical-like)")
        
        # 6. Energy distribution
        low_freq_ratio = features.get('low_freq_ratio', 0)
        high_freq_ratio = features.get('high_freq_ratio', 0)
        
        # Traffic/machinery has more low frequency energy
        if low_freq_ratio > 0.6:
            score -= 0.2
            confidence_factors.append("High low-freq energy (traffic-like)")
        
        # Birds often have good high frequency content
        if high_freq_ratio > 0.2:
            score += 0.1
            confidence_factors.append("Good high-freq content")
        
        # 7. Spectral bandwidth
        bandwidth = features.get('spectral_bandwidth_mean', 0)
        if 1000 <= bandwidth <= 3000:
            score += 0.05
            confidence_factors.append("Good spectral bandwidth")
        
        # Clamp score to [0, 1]
        score = max(0, min(1, score))
        
        return score, confidence_factors


class AudioFilter:
    """Main audio filtering class."""
    
    def __init__(self, input_dir, output_dir=None, action='move', threshold=0.6):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else None
        self.action = action  # 'move', 'delete', 'report'
        self.threshold = threshold
        self.classifier = AudioClassifier()
        
        # Create output directories
        if self.action == 'move' and self.output_dir:
            self.filtered_dir = self.output_dir / 'crow_sounds'
            self.rejected_dir = self.output_dir / 'non_crow_sounds'
            self.filtered_dir.mkdir(parents=True, exist_ok=True)
            self.rejected_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'crow_sounds': 0,
            'non_crow_sounds': 0,
            'errors': 0
        }
        
        # Results log
        self.results = []
    
    def find_audio_files(self):
        """Find all audio files in the input directory."""
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(self.input_dir.rglob(f'*{ext}'))
        
        return sorted(audio_files)
    
    def process_file(self, audio_path):
        """Process a single audio file."""
        try:
            # Extract features
            features = self.classifier.extract_features(audio_path)
            if features is None:
                return None
            
            # Classify
            score, factors = self.classifier.classify_audio(features)
            
            # Determine classification
            is_crow = score >= self.threshold
            
            result = {
                'file': str(audio_path),
                'score': score,
                'is_crow': is_crow,
                'confidence_factors': factors,
                'relative_path': str(audio_path.relative_to(self.input_dir))
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            return None
    
    def execute_action(self, result):
        """Execute the specified action on a file based on classification."""
        audio_path = Path(result['file'])
        is_crow = result['is_crow']
        
        if self.action == 'move' and self.output_dir:
            if is_crow:
                dest_dir = self.filtered_dir
                dest_path = dest_dir / result['relative_path']
            else:
                dest_dir = self.rejected_dir
                dest_path = dest_dir / result['relative_path']
            
            # Create parent directory
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.move(str(audio_path), str(dest_path))
            logger.debug(f"Moved {audio_path.name} to {dest_dir.name}")
            
        elif self.action == 'delete' and not is_crow:
            audio_path.unlink()
            logger.debug(f"Deleted {audio_path.name}")
    
    def process_directory(self):
        """Process all audio files in the directory."""
        audio_files = self.find_audio_files()
        self.stats['total_files'] = len(audio_files)
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        for audio_path in tqdm(audio_files, desc="Processing audio files"):
            result = self.process_file(audio_path)
            
            if result is None:
                self.stats['errors'] += 1
                continue
            
            self.results.append(result)
            self.stats['processed'] += 1
            
            if result['is_crow']:
                self.stats['crow_sounds'] += 1
            else:
                self.stats['non_crow_sounds'] += 1
            
            # Execute action
            if self.action != 'report':
                self.execute_action(result)
    
    def save_report(self, report_path=None):
        """Save detailed filtering report."""
        if report_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.input_dir / f"audio_filter_report_{timestamp}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'input_directory': str(self.input_dir),
            'output_directory': str(self.output_dir) if self.output_dir else None,
            'action': self.action,
            'threshold': self.threshold,
            'statistics': self.stats,
            'results': self.results
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved detailed report to: {report_path}")
    
    def print_summary(self):
        """Print filtering summary."""
        print(f"\n{'='*60}")
        print(f"AUDIO FILTERING SUMMARY")
        print(f"{'='*60}")
        print(f"Input directory: {self.input_dir}")
        print(f"Threshold: {self.threshold}")
        print(f"Action: {self.action}")
        print(f"\nResults:")
        print(f"  Total files found: {self.stats['total_files']}")
        print(f"  Successfully processed: {self.stats['processed']}")
        print(f"  Classified as crow sounds: {self.stats['crow_sounds']}")
        print(f"  Classified as non-crow sounds: {self.stats['non_crow_sounds']}")
        print(f"  Errors: {self.stats['errors']}")
        
        if self.stats['processed'] > 0:
            crow_percentage = (self.stats['crow_sounds'] / self.stats['processed']) * 100
            print(f"  Crow sound percentage: {crow_percentage:.1f}%")
        
        # Show some examples
        if self.results:
            print(f"\nTop 5 most confident crow sounds:")
            crow_results = [r for r in self.results if r['is_crow']]
            crow_results.sort(key=lambda x: x['score'], reverse=True)
            for result in crow_results[:5]:
                print(f"  {Path(result['file']).name}: {result['score']:.3f}")
            
            print(f"\nTop 5 most confident non-crow sounds:")
            non_crow_results = [r for r in self.results if not r['is_crow']]
            non_crow_results.sort(key=lambda x: x['score'])
            for result in non_crow_results[:5]:
                print(f"  {Path(result['file']).name}: {result['score']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter crow audio training data to remove non-crow sounds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Just analyze and report (no file changes)
  python audio_filter.py /path/to/audio --action report
  
  # Move files to organized directories
  python audio_filter.py /path/to/audio --output-dir filtered_audio --action move
  
  # Delete non-crow sounds (be careful!)
  python audio_filter.py /path/to/audio --action delete --threshold 0.8
  
  # More aggressive filtering
  python audio_filter.py /path/to/audio --threshold 0.7 --action move
        """
    )
    
    parser.add_argument('input_dir', 
                       help='Directory containing audio files to filter')
    
    parser.add_argument('--output-dir', 
                       help='Output directory for filtered files (required for move action)')
    
    parser.add_argument('--action', 
                       choices=['report', 'move', 'delete'],
                       default='report',
                       help='Action to take: report only, move files, or delete non-crow files')
    
    parser.add_argument('--threshold',
                       type=float,
                       default=0.6,
                       help='Confidence threshold for crow classification (0.0-1.0, default: 0.6)')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    if args.action == 'move' and not args.output_dir:
        print("Error: --output-dir is required when using --action move")
        sys.exit(1)
    
    if not 0.0 <= args.threshold <= 1.0:
        print("Error: Threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Confirm destructive actions
    if args.action == 'delete':
        response = input(f"WARNING: This will permanently DELETE audio files classified as non-crow sounds.\n"
                        f"Threshold: {args.threshold}\n"
                        f"Directory: {args.input_dir}\n"
                        f"Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            sys.exit(0)
    
    # Run filtering
    logger.info(f"Starting audio filtering with threshold {args.threshold}")
    filter_tool = AudioFilter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        action=args.action,
        threshold=args.threshold
    )
    
    filter_tool.process_directory()
    filter_tool.print_summary()
    filter_tool.save_report()
    
    logger.info("Audio filtering complete!")


if __name__ == "__main__":
    main() 