import pytest
import torch
import numpy as np
import random
from model import AudioFeatureExtractor, CrowResNetEmbedder, CrowMultiModalEmbedder
from models import CrowResNetEmbedder as NewCrowResNetEmbedder, create_model

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

def test_new_crow_resnet_embedder_512d(device):
    """Test new CrowResNetEmbedder with 512D embeddings."""
    # Test default 512D embeddings
    model = NewCrowResNetEmbedder(embedding_dim=512, device=device)
    
    # Test with dummy input
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    output = model(x)
    
    # Check output shape and properties
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 512)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    
    # Test normalization (should be L2 normalized)
    norms = torch.norm(output, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_new_crow_resnet_embedder_configurable_dims(device):
    """Test new CrowResNetEmbedder with different embedding dimensions."""
    test_dims = [128, 256, 512, 1024]
    
    for dim in test_dims:
        model = NewCrowResNetEmbedder(embedding_dim=dim, device=device)
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224).to(device)
        output = model(x)
        
        assert output.shape == (batch_size, dim)
        assert not torch.isnan(output).any()
        
        # Test L2 normalization
        norms = torch.norm(output, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_create_model_function(device):
    """Test the create_model factory function."""
    # Test default parameters
    model = create_model(device=device)
    assert isinstance(model, NewCrowResNetEmbedder)
    
    # Test with custom embedding dimension
    model_256 = create_model(embedding_dim=256, device=device)
    model_512 = create_model(embedding_dim=512, device=device)
    
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    
    output_256 = model_256(x)
    output_512 = model_512(x)
    
    assert output_256.shape == (batch_size, 256)
    assert output_512.shape == (batch_size, 512)

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

def test_crow_multi_modal_embedder_512d():
    """Test multi-modal embedder with 512D final embeddings."""
    torch.manual_seed(0)
    np.random.seed(0)
    model = CrowMultiModalEmbedder(
        visual_embedding_dim=512,
        audio_embedding_dim=512,
        final_embedding_dim=512  # Test 512D output
    )
    
    # Create dummy input
    batch_size = 3
    imgs = torch.randn(batch_size, 3, 224, 224)
    audio = {
        'mel_spec': torch.randn(batch_size, 128, 64),
        'chroma': torch.randn(batch_size, 12, 64)
    }
    
    # Test forward pass
    output = model(imgs, audio)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 512)
    
    # Test L2 normalization
    norms = torch.norm(output, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

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
    """Test model saving and loading with 512D embeddings."""
    torch.manual_seed(0)
    np.random.seed(0)
    model = CrowMultiModalEmbedder(
        visual_embedding_dim=512,
        audio_embedding_dim=512,
        final_embedding_dim=512  # Test 512D
    )
    save_path = tmp_path / "model_512d.pt"
    torch.save(model.state_dict(), save_path)
    
    # Load model
    loaded_model = CrowMultiModalEmbedder(
        visual_embedding_dim=512,
        audio_embedding_dim=512,
        final_embedding_dim=512
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
        final_embedding_dim=512  # Test 512D consistency
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

def test_embedding_similarity_properties():
    """Test that similar inputs produce similar embeddings."""
    model = NewCrowResNetEmbedder(embedding_dim=512)
    model.eval()
    
    # Create base image
    base_img = torch.randn(1, 3, 224, 224)
    
    # Create slightly modified version
    noise = torch.randn_like(base_img) * 0.1  # Small noise
    similar_img = base_img + noise
    
    # Create very different image
    different_img = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        base_emb = model(base_img)
        similar_emb = model(similar_img)
        different_emb = model(different_img)
        
        # Similar images should have higher cosine similarity
        sim_similar = torch.cosine_similarity(base_emb, similar_emb, dim=1)
        sim_different = torch.cosine_similarity(base_emb, different_emb, dim=1)
        
        # This is a weak test but checks basic sanity
        assert sim_similar.item() > 0.5  # Should be reasonably similar
        assert abs(sim_different.item()) < abs(sim_similar.item())  # Should be less similar

def test_batch_size_independence():
    """Test that embedding is independent of batch size."""
    model = NewCrowResNetEmbedder(embedding_dim=512)
    model.eval()
    
    # Create test images
    img1 = torch.randn(1, 3, 224, 224)
    img2 = torch.randn(1, 3, 224, 224)
    
    # Test individual processing
    with torch.no_grad():
        emb1_single = model(img1)
        emb2_single = model(img2)
        
        # Test batch processing
        batch_imgs = torch.cat([img1, img2], dim=0)
        batch_embs = model(batch_imgs)
        
        # Results should be identical
        assert torch.allclose(emb1_single, batch_embs[0:1], atol=1e-6)
        assert torch.allclose(emb2_single, batch_embs[1:2], atol=1e-6)

def test_device_handling():
    """Test proper device handling for models."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Test CPU model
    model_cpu = NewCrowResNetEmbedder(embedding_dim=512, device=torch.device('cpu'))
    assert next(model_cpu.parameters()).device.type == 'cpu'
    
    # Test GPU model
    model_gpu = NewCrowResNetEmbedder(embedding_dim=512, device=torch.device('cuda'))
    assert next(model_gpu.parameters()).device.type == 'cuda'
    
    # Test input processing
    x_cpu = torch.randn(1, 3, 224, 224)
    x_gpu = x_cpu.cuda()
    
    output_cpu = model_cpu(x_cpu)
    output_gpu = model_gpu(x_gpu)
    
    assert output_cpu.device.type == 'cpu'
    assert output_gpu.device.type == 'cuda'
    
    # Results should be similar (allowing for floating point differences)
    assert torch.allclose(output_cpu, output_gpu.cpu(), atol=1e-4) 