import pytest
import torch
import numpy as np
import random
from model import AudioFeatureExtractor, CrowResNetEmbedder, CrowMultiModalEmbedder

def test_audio_feature_extractor():
    """Test audio feature extractor."""
    torch.manual_seed(0)
    np.random.seed(0)
    model = AudioFeatureExtractor(mel_dim=128, chroma_dim=12)
    
    # Create dummy input
    batch_size = 2
    mel_spec = torch.randn(batch_size, 128, 64)  # (batch, mel_bins, time)
    chroma = torch.randn(batch_size, 12, 64)  # (batch, chroma_bins, time)
    x = {'mel_spec': mel_spec, 'chroma': chroma}
    
    # Test forward pass
    output = model(x)
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == batch_size
    assert output.shape[1] == model.embedding_dim

def test_crow_resnet_embedder(device):
    """Test visual feature extractor."""
    model = CrowResNetEmbedder().to(device)
    
    # Test with dummy input
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224).to(device)  # RGB image
    output = model(x)
    
    # Check output
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 512)  # output_dim=512
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    
    # Test with invalid input
    with pytest.raises(ValueError):
        model(torch.randn(batch_size, 1, 224, 224).to(device))  # Wrong number of channels

def test_crow_multi_modal_embedder():
    """Test multi-modal embedder."""
    torch.manual_seed(0)
    np.random.seed(0)
    model = CrowMultiModalEmbedder(
        visual_embedding_dim=512,
        audio_embedding_dim=256,
        final_embedding_dim=128
    )
    
    # Create dummy input
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 224, 224)  # RGB images
    audio = {
        'mel_spec': torch.randn(batch_size, 128, 64),  # Mel spectrogram
        'chroma': torch.randn(batch_size, 12, 64)  # Chroma features
    }
    
    # Test forward pass
    output = model(imgs, audio)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, model.final_embedding_dim)
    
    # Test with audio only
    output_audio = model(None, audio)
    assert isinstance(output_audio, torch.Tensor)
    assert output_audio.shape == (batch_size, model.final_embedding_dim)
    
    # Test with image only
    output_img = model(imgs, None)
    assert isinstance(output_img, torch.Tensor)
    assert output_img.shape == (batch_size, model.final_embedding_dim)

def test_model_gradients():
    """Test model gradients."""
    model = CrowMultiModalEmbedder(
        visual_embedding_dim=512,
        audio_embedding_dim=256,
        final_embedding_dim=128
    )
    
    # Create dummy input
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
    audio = {
        'mel_spec': torch.randn(batch_size, 128, 64, requires_grad=True),
        'chroma': torch.randn(batch_size, 12, 64, requires_grad=True)
    }
    
    # Forward pass
    output = model(imgs, audio)
    
    # Compute loss and backward pass
    loss = output.mean()
    loss.backward()
    
    # Check gradients
    assert imgs.grad is not None
    assert audio['mel_spec'].grad is not None
    assert audio['chroma'].grad is not None
    assert torch.any(imgs.grad != 0)
    assert torch.any(audio['mel_spec'].grad != 0)
    assert torch.any(audio['chroma'].grad != 0)

def test_model_save_load(tmp_path):
    """Test model saving and loading."""
    torch.manual_seed(0)
    np.random.seed(0)
    model = CrowMultiModalEmbedder(
        visual_embedding_dim=512,
        audio_embedding_dim=256,
        final_embedding_dim=128
    )
    save_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), save_path)
    
    # Load model
    loaded_model = CrowMultiModalEmbedder(
        visual_embedding_dim=512,
        audio_embedding_dim=256,
        final_embedding_dim=128
    )
    loaded_model.load_state_dict(torch.load(save_path))
    
    model.eval()
    loaded_model.eval()
    
    # Create dummy input
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 224, 224)
    audio = {
        'mel_spec': torch.randn(batch_size, 128, 64),
        'chroma': torch.randn(batch_size, 12, 64)
    }
    
    # Compare outputs
    with torch.no_grad():
        output1 = model(imgs, audio)
        output2 = loaded_model(imgs, audio)
        assert torch.allclose(output1, output2, atol=1e-5)

def test_model_consistency():
    """Test model consistency across different inputs."""
    model = CrowMultiModalEmbedder(
        visual_embedding_dim=512,
        audio_embedding_dim=256,
        final_embedding_dim=128
    )
    model.eval()
    
    # Create dummy input
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 224, 224)
    audio = {
        'mel_spec': torch.randn(batch_size, 128, 64),
        'chroma': torch.randn(batch_size, 12, 64)
    }
    
    # Test consistency
    with torch.no_grad():
        output1 = model(imgs, audio)
        output2 = model(imgs, audio)
        assert torch.allclose(output1, output2)
        
        # Test with different audio lengths
        audio_long = {
            'mel_spec': torch.randn(batch_size, 128, 128),
            'chroma': torch.randn(batch_size, 12, 128)
        }
        output3 = model(imgs, audio_long)
        assert output3.shape == output1.shape 