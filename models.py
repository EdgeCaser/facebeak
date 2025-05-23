import torch
import torch.nn as nn
import torchvision
import logging
import torchvision.models as models

class AudioFeatureExtractor(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=256):
        """Extract features from audio spectrograms."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * (input_dim//8) * 12, hidden_dim)  # 12 is for chroma features
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # x shape: (batch_size, 2, n_mels, time) - 2 channels for mel_spec and chroma
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CrowMultiModalEmbedder(nn.Module):
    def __init__(self, visual_embed_dim=512, audio_embed_dim=256, final_embed_dim=512, device=None):
        """
        Multi-modal model for crow identification using both visual and audio features.
        
        Args:
            visual_embed_dim: Dimension of visual embeddings
            audio_embed_dim: Dimension of audio embeddings
            final_embed_dim: Dimension of final combined embeddings
            device: Device to place model on (if None, will use CUDA if available)
        """
        super().__init__()
        
        # Determine device with improved logging
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                torch.cuda.empty_cache()  # Clear any existing allocations
                logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device('cpu')
                logging.info("CUDA not available, using CPU")
        self.device = device
        
        # Visual feature extractor (ResNet)
        try:
            base = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            self.visual_extractor = nn.Sequential(*list(base.children())[:-1])
            self.visual_fc = nn.Linear(512, visual_embed_dim)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize visual extractor: {str(e)}")
        
        # Audio feature extractor
        try:
            self.audio_extractor = AudioFeatureExtractor(output_dim=audio_embed_dim)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize audio extractor: {str(e)}")
        
        # Fusion layer
        try:
            self.fusion = nn.Sequential(
                nn.Linear(visual_embed_dim + audio_embed_dim, final_embed_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(final_embed_dim, final_embed_dim)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize fusion layer: {str(e)}")
        
        # Move model to device and verify
        self._move_to_device()
        
    def _move_to_device(self):
        """Move model to device and verify all parameters are on the correct device."""
        self.to(self.device)
        
        # Verify model parameters are on correct device
        param_devices = set(str(p.device) for p in self.parameters())
        expected_device = str(self.device)
        
        # More flexible device checking for CUDA
        if expected_device.startswith('cuda'):
            if not all(d.startswith('cuda') for d in param_devices):
                raise RuntimeError(f"Some model parameters not on CUDA device")
        else:
            if not all(d == expected_device for d in param_devices):
                raise RuntimeError(f"Model parameters not on expected device {self.device}")
        
        logging.info(f"Model successfully moved to {self.device}")
        
    def forward(self, visual_input, audio_input=None):
        """Forward pass with improved device handling."""
        # Handle None inputs
        if visual_input is None and audio_input is None:
            raise ValueError("At least one of visual_input or audio_input must be provided")
            
        # Process visual input if available
        if visual_input is not None:
            # Only move to device if not already there
            if visual_input.device != self.device:
                visual_input = visual_input.to(self.device)
            visual_features = self.visual_extractor(visual_input)
            visual_features = visual_features.view(visual_features.size(0), -1)
            visual_features = self.visual_fc(visual_features)
        else:
            visual_features = None
        
        # Process audio input if available
        if audio_input is not None:
            # Move audio inputs to device only if needed
            if isinstance(audio_input, dict):
                audio_input = {k: v.to(self.device) if v.device != self.device else v 
                             for k, v in audio_input.items()}
            else:
                if audio_input.device != self.device:
                    audio_input = audio_input.to(self.device)
                    
            audio_features = self.audio_extractor(audio_input)
            
            if visual_features is not None:
                # Combine features
                combined = torch.cat([visual_features, audio_features], dim=1)
                embeddings = self.fusion(combined)
            else:
                embeddings = audio_features
        else:
            embeddings = visual_features
            
        # L2 normalize for triplet loss
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

class CrowResNetEmbedder(nn.Module):
    def __init__(self, embedding_dim=512, device=None):
        """
        ResNet-based embedder for crow identification.
        
        Args:
            embedding_dim: Dimension of output embeddings
            device: Device to place model on (if None, will use CUDA if available)
        """
        super().__init__()
        
        # Determine device with improved logging
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                torch.cuda.empty_cache()  # Clear any existing allocations
                logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device('cpu')
                logging.info("CUDA not available, using CPU")
        self.device = device
        
        try:
            base = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1])  # Remove classifier
            self.fc = nn.Linear(512, embedding_dim)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ResNet embedder: {str(e)}")
        
        # Move model to device and verify
        self._move_to_device()
        
    def _move_to_device(self):
        """Move model to device and verify all parameters are on the correct device."""
        self.to(self.device)
        
        # Verify model parameters are on correct device
        param_devices = set(str(p.device) for p in self.parameters())
        expected_device = str(self.device)
        
        # More flexible device checking for CUDA
        if expected_device.startswith('cuda'):
            if not all(d.startswith('cuda') for d in param_devices):
                raise RuntimeError(f"Some model parameters not on CUDA device")
        else:
            if not all(d == expected_device for d in param_devices):
                raise RuntimeError(f"Model parameters not on expected device {self.device}")
        
        logging.info(f"Model successfully moved to {self.device}")
        
    def forward(self, x):
        """Forward pass with improved device handling."""
        # Only move to device if not already there
        if x.device != self.device:
            x = x.to(self.device)
        
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)  # L2 normalize for triplet loss
        return x 

def create_model(embedding_dim=512, device=None):
    """Create and return a ResNet18 model for feature extraction.
    
    Args:
        embedding_dim: Dimension of the output embedding
        device: Device to place the model on
        
    Returns:
        CrowResNetEmbedder: Model for feature extraction
    """
    try:
        model = CrowResNetEmbedder(embedding_dim=embedding_dim, device=device)
        if device is not None:
            model = model.to(device)
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        logging.error(f"Failed to create model: {str(e)}")
        raise 