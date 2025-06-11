import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import logging
from audio import extract_audio_features
import torchvision.transforms as T
from pathlib import Path
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class CrowTripletDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """Initialize dataset.
        
        Args:
            data_dir: Path to data directory
            transform: Optional transform to apply to images
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Default transform if none provided
        if transform is None:
            self.transform = T.Compose([
                T.Resize((580, 580)),  # Resize to larger square to preserve aspect ratio
                T.CenterCrop((512, 512)),  # Crop to target size
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Find all crow directories
        self.crow_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        if not self.crow_dirs:
            raise ValueError(f"No crow directories found in {data_dir}")
        
        # Initialize data structures
        self.crow_to_images = {}  # crow_id -> list of image paths
        self.crow_to_audio = {}   # crow_id -> list of audio paths
        self.crow_to_videos = {}  # crow_id -> list of video paths
        self.samples = []         # list of (image_path, video_path, audio_path, crow_id) tuples
        
        # Load data
        for crow_dir in self.crow_dirs:
            crow_id = crow_dir.name
            
            # Find images
            image_dir = crow_dir / 'images'
            if image_dir.exists():
                image_paths = sorted(image_dir.glob('*.jpg')) + sorted(image_dir.glob('*.jpeg'))
                if image_paths:
                    self.crow_to_images[crow_id] = image_paths
                else:
                    logger.warning(f"No images found for crow {crow_id}")
            else:
                logger.warning(f"No images found for crow {crow_id}")
            
            # Find video files
            video_dir = crow_dir / 'videos'
            if video_dir.exists():
                video_paths = sorted(video_dir.glob('*.mp4'))
                if video_paths:
                    self.crow_to_videos[crow_id] = video_paths
                else:
                    logger.warning(f"No video files found for crow {crow_id}")
            else:
                logger.warning(f"No video files found for crow {crow_id}")
            
            # Find audio files
            audio_dir = crow_dir / 'audio'
            if audio_dir.exists():
                audio_paths = sorted(audio_dir.glob('*.wav'))
                if audio_paths:
                    self.crow_to_audio[crow_id] = audio_paths
                else:
                    logger.warning(f"No audio files found for crow {crow_id}")
            else:
                logger.warning(f"No audio files found for crow {crow_id}")
            
            # Create samples
            if crow_id in self.crow_to_images:
                for img_path in self.crow_to_images[crow_id]:
                    # Find matching video file if available
                    video_path = None
                    if crow_id in self.crow_to_videos:
                        # Try to find video file with same name
                        video_name = img_path.stem + '.mp4'
                        matching_video = [p for p in self.crow_to_videos[crow_id] if p.name == video_name]
                        if matching_video:
                            video_path = matching_video[0]
                        else:
                            # If no matching video, use first available
                            video_path = self.crow_to_videos[crow_id][0]
                    
                    # Find matching audio file if available
                    audio_path = None
                    if crow_id in self.crow_to_audio:
                        # Try to find audio file with same name
                        audio_name = img_path.stem + '.wav'
                        matching_audio = [p for p in self.crow_to_audio[crow_id] if p.name == audio_name]
                        if matching_audio:
                            audio_path = matching_audio[0]
                        else:
                            # If no matching audio, use first available
                            audio_path = self.crow_to_audio[crow_id][0]
                    
                    self.samples.append((img_path, video_path, audio_path, crow_id))
        
        if not self.samples:
            raise ValueError("No valid samples found in dataset")
        
        logger.info(f"Initialized dataset with {len(self.samples)} samples from {len(self.crow_dirs)} crows")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx: Index of sample to get
            
        Returns:
            dict: Sample containing:
                - image: torch.Tensor of shape (3, 224, 224)
                - video_path: Path to video file (or None)
                - audio: dict containing:
                    - mel_spec: torch.Tensor of shape (128, time)
                    - chroma: torch.Tensor of shape (12, time)
                - crow_id: str
        """
        img_path, video_path, audio_path, crow_id = self.samples[idx]
        
        # Load and transform image
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            else:
                # Apply default transforms if none provided
                image = transforms.ToTensor()(image)
                image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])(image)
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise
        
        # Initialize audio features as None
        audio = None
        
        # Load audio features if available
        if audio_path is not None and audio_path.exists():
            try:
                # Extract audio features
                mel_spec, chroma = extract_audio_features(str(audio_path))
                
                # Convert to tensors and ensure proper types
                mel_spec = torch.from_numpy(mel_spec).float()
                chroma = torch.from_numpy(chroma).float()
                
                # Ensure proper shapes and normalization
                if mel_spec.ndim == 2:
                    mel_spec = mel_spec.unsqueeze(0)  # Add batch dimension
                if chroma.ndim == 2:
                    chroma = chroma.unsqueeze(0)  # Add batch dimension
                
                # Normalize features
                mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
                chroma = (chroma - chroma.mean()) / (chroma.std() + 1e-6)
                
                # Create audio dictionary
                audio = {
                    'mel_spec': mel_spec,
                    'chroma': chroma
                }
                
            except Exception as e:
                logger.warning(f"Error loading audio features from {audio_path}: {e}")
                # Don't raise the error, just keep audio as None
        
        # Create sample dictionary
        sample = {
            'image': image,
            'video_path': str(video_path) if video_path else None,
            'audio': audio,
            'crow_id': crow_id
        }
        
        return sample
    
    def collate_fn(self, batch):
        """Collate function for DataLoader.
        
        Args:
            batch: List of samples from __getitem__
            
        Returns:
            dict: Batched samples containing:
                - image: torch.Tensor of shape (batch_size, 3, 224, 224)
                - video_paths: list of Path objects
                - audio: dict containing:
                    - mel_spec: torch.Tensor of shape (batch_size, 128, max_time)
                    - chroma: torch.Tensor of shape (batch_size, 12, max_time)
                - crow_id: list of str
        """
        # Separate images, audio, and labels
        images = [sample['image'] for sample in batch]
        video_paths = [sample['video_path'] for sample in batch]
        audio_dicts = [sample['audio'] for sample in batch]
        crow_ids = [sample['crow_id'] for sample in batch]
        
        # Stack images
        images = torch.stack(images)
        
        # Handle audio features
        if any(audio is not None for audio in audio_dicts):
            # Get max time dimension
            max_time = max(
                max(audio['mel_spec'].shape[1], audio['chroma'].shape[1])
                for audio in audio_dicts if audio is not None
            )
            
            # Pad and stack mel spectrograms
            mel_specs = []
            for audio in audio_dicts:
                if audio is not None:
                    mel_spec = audio['mel_spec']
                    if mel_spec.shape[1] < max_time:
                        mel_spec = F.pad(mel_spec, (0, max_time - mel_spec.shape[1]))
                    mel_specs.append(mel_spec)
                else:
                    mel_specs.append(torch.zeros(128, max_time))
            mel_specs = torch.stack(mel_specs)
            
            # Pad and stack chroma features
            chromas = []
            for audio in audio_dicts:
                if audio is not None:
                    chroma = audio['chroma']
                    if chroma.shape[1] < max_time:
                        chroma = F.pad(chroma, (0, max_time - chroma.shape[1]))
                    chromas.append(chroma)
                else:
                    chromas.append(torch.zeros(12, max_time))
            chromas = torch.stack(chromas)
            
            audio = {
                'mel_spec': mel_specs,
                'chroma': chromas
            }
        else:
            audio = None
        
        return {
            'image': images,
            'video_paths': video_paths,
            'audio': audio,
            'crow_id': crow_ids
        }
    
    @property
    def crow_ids(self):
        """Get list of unique crow IDs."""
        return sorted(set(crow_id for _, _, _, crow_id in self.samples)) 