import torch
import torch.nn as nn
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)

class AudioFeatureExtractor(nn.Module):
    def __init__(self, mel_dim=128, chroma_dim=12, hidden_dim=256, output_dim=512):
        """
        Audio feature extractor using CNN.
        
        Args:
            mel_dim: Number of mel bands
            chroma_dim: Number of chroma bins
            hidden_dim: Hidden dimension size
            output_dim: Output embedding dimension
        """
        super().__init__()
        
        # Mel spectrogram branch
        self.mel_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Chroma branch
        self.chroma_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Calculate input size for FC layers
        mel_size = self._get_conv_output_size(mel_dim)
        chroma_size = self._get_conv_output_size(chroma_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(mel_size + chroma_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )
        self.embedding_dim = output_dim
    
    def _get_conv_output_size(self, input_size):
        """Calculate the output size after conv layers."""
        x = torch.randn(1, 1, input_size, 100)  # Assume 100 time steps
        x = self.mel_conv(x)  # Use mel_conv as reference
        x = self.adaptive_pool(x)
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Dictionary containing 'mel_spec' and 'chroma' tensors
               Each tensor should be of shape (batch_size, height, width)
        """
        if not isinstance(x, dict) or 'mel_spec' not in x or 'chroma' not in x:
            raise ValueError("Input must be a dictionary with 'mel_spec' and 'chroma' keys")
        
        # Add channel dimension if needed
        mel_spec = x['mel_spec'].unsqueeze(1) if len(x['mel_spec'].shape) == 3 else x['mel_spec']
        chroma = x['chroma'].unsqueeze(1) if len(x['chroma'].shape) == 3 else x['chroma']
        
        # Process mel spectrogram
        mel_features = self.mel_conv(mel_spec)
        mel_features = self.adaptive_pool(mel_features)
        mel_features = mel_features.view(mel_features.size(0), -1)
        
        # Process chroma
        chroma_features = self.chroma_conv(chroma)
        chroma_features = self.adaptive_pool(chroma_features)
        chroma_features = chroma_features.view(chroma_features.size(0), -1)
        
        # Concatenate and project
        combined = torch.cat([mel_features, chroma_features], dim=1)
        return self.fc(combined)

class CrowResNetEmbedder(nn.Module):
    def __init__(self, pretrained=True, output_dim=512):
        """
        Visual feature extractor using ResNet.
        
        Args:
            pretrained: Whether to use pretrained weights
            output_dim: Output embedding dimension
        """
        super().__init__()
        
        # Load pretrained ResNet
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add new FC layer for embedding
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        """
        # Ensure input has correct shape
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width), got {len(x.shape)}D")
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 input channels, got {x.shape[1]}")
        
        # Extract features
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        # Project to embedding space
        x = self.fc(x)
        
        return x

class CrowMultiModalEmbedder(nn.Module):
    def __init__(self, visual_dim=None, audio_dim=None, hidden_dim=1024, output_dim=512,
                 visual_embed_dim=None, audio_embed_dim=None, final_embed_dim=None, device=None, **kwargs):
        """
        Multi-modal embedder combining visual and audio features.
        Accepts both (visual_dim, audio_dim, output_dim) and
        (visual_embed_dim, audio_embed_dim, final_embed_dim) for compatibility.
        
        Args:
            visual_dim/visual_embed_dim: Dimension of visual embeddings
            audio_dim/audio_embed_dim: Dimension of audio embeddings
            hidden_dim: Dimension of hidden layers
            output_dim/final_embed_dim: Dimension of final combined embeddings
            device: Device to place model on (if None, will use CUDA if available)
        """
        super().__init__()
        
        # Backward compatibility
        if visual_embed_dim is not None:
            visual_dim = visual_embed_dim
        if audio_embed_dim is not None:
            audio_dim = audio_embed_dim
        if final_embed_dim is not None:
            output_dim = final_embed_dim

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

        # Initialize components
        try:
            self.visual_embedder = CrowResNetEmbedder(output_dim=visual_dim or 512)
            self.audio_embedder = AudioFeatureExtractor(output_dim=audio_dim or 512)
            
            # Fusion layers
            self.fusion = nn.Sequential(
                nn.Linear((visual_dim or 512) + (audio_dim or 512), hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, output_dim)
            )
            
            # Visual-only projection
            self.visual_proj = nn.Sequential(
                nn.Linear(visual_dim or 512, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, output_dim)
            )
            self.final_embedding_dim = output_dim
            
            # Move model to device and verify
            self._move_to_device()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model components: {str(e)}")
    
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
        """Forward pass with improved device handling and input validation.
        
        Args:
            visual_input: Image tensor of shape (batch_size, 3, height, width) or None
            audio_input: Optional dictionary containing 'mel_spec' and 'chroma' tensors or None
            
        Returns:
            torch.Tensor: Combined embeddings of shape (batch_size, final_embed_dim)
        """
        # Validate inputs
        if visual_input is None and audio_input is None:
            raise ValueError("At least one of visual_input or audio_input must be provided")
            
        # Process visual input if available
        if visual_input is not None:
            # Ensure proper type and device
            if not isinstance(visual_input, torch.Tensor):
                raise TypeError("visual_input must be a torch.Tensor")
            if visual_input.device != self.device:
                visual_input = visual_input.to(self.device)
                
            # Process visual features
            visual_features = self.visual_embedder(visual_input)
            visual_features = visual_features.view(visual_features.size(0), -1)
            visual_features = self.visual_proj(visual_features)
        else:
            visual_features = None
        
        # Process audio input if available
        if audio_input is not None:
            # Validate audio input structure
            if not isinstance(audio_input, dict):
                raise TypeError("audio_input must be a dictionary")
            if 'mel_spec' not in audio_input or 'chroma' not in audio_input:
                raise ValueError("audio_input must contain 'mel_spec' and 'chroma' keys")
                
            # Ensure proper types and device for audio features
            mel_spec = audio_input['mel_spec']
            chroma = audio_input['chroma']
            
            if not isinstance(mel_spec, torch.Tensor) or not isinstance(chroma, torch.Tensor):
                raise TypeError("mel_spec and chroma must be torch.Tensors")
                
            if mel_spec.device != self.device:
                mel_spec = mel_spec.to(self.device)
            if chroma.device != self.device:
                chroma = chroma.to(self.device)
                
            # Process audio features
            audio_features = self.audio_embedder({
                'mel_spec': mel_spec,
                'chroma': chroma
            })
            
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
    
    def get_visual_embedding(self, image):
        """Get visual-only embedding."""
        return self.visual_embedder(image)
    
    def get_audio_embedding(self, audio):
        """Get audio-only embedding."""
        return self.audio_embedder(audio) 