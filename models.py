import torch
import torch.nn as nn
import torchvision

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
    def __init__(self, visual_embed_dim=512, audio_embed_dim=256, final_embed_dim=512):
        """
        Multi-modal model for crow identification using both visual and audio features.
        
        Args:
            visual_embed_dim: Dimension of visual embeddings
            audio_embed_dim: Dimension of audio embeddings
            final_embed_dim: Dimension of final combined embeddings
        """
        super().__init__()
        # Visual feature extractor (ResNet)
        base = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.visual_extractor = nn.Sequential(*list(base.children())[:-1])
        self.visual_fc = nn.Linear(512, visual_embed_dim)
        
        # Audio feature extractor
        self.audio_extractor = AudioFeatureExtractor(output_dim=audio_embed_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(visual_embed_dim + audio_embed_dim, final_embed_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(final_embed_dim, final_embed_dim)
        )
        
    def forward(self, visual_input, audio_input=None):
        # Visual features
        visual_features = self.visual_extractor(visual_input)
        visual_features = visual_features.view(visual_features.size(0), -1)
        visual_features = self.visual_fc(visual_features)
        
        if audio_input is not None:
            # Audio features
            audio_features = self.audio_extractor(audio_input)
            
            # Combine features
            combined = torch.cat([visual_features, audio_features], dim=1)
            embeddings = self.fusion(combined)
        else:
            # If no audio input, just use visual features
            embeddings = visual_features
            
        # L2 normalize for triplet loss
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

# Keep the original CrowResNetEmbedder for backward compatibility
class CrowResNetEmbedder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        base = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])  # Remove classifier
        self.fc = nn.Linear(512, embedding_dim)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)  # L2 normalize for triplet loss
        return x 