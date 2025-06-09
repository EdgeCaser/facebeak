import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

# Add project root to sys.path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilities.simple_evaluate import evaluate_model # Assuming path
# from utilities.simple_evaluate import load_sample_data # To be mocked
# from utilities.simple_evaluate import load_model_correct # To be mocked

# Dummy embeddings and labels for testing
# Format: [embedding_vector, label_string]
# Crow embeddings (vectors are just placeholders, distances matter)
dummy_crow_emb1_idA = np.array([0.1, 0.2, 0.3, 0.4]) # Crow A, Sample 1
dummy_crow_emb2_idA = np.array([0.12, 0.22, 0.32, 0.42]) # Crow A, Sample 2 (similar to 1)
dummy_crow_emb1_idB = np.array([0.8, 0.7, 0.6, 0.5]) # Crow B, Sample 1
dummy_crow_emb2_idB = np.array([0.81, 0.71, 0.61, 0.51]) # Crow B, Sample 2 (similar to B1)

# Non-crow embeddings (should be distinct from crows and each other ideally for some tests)
dummy_noncrow_emb1 = np.array([-0.5, -0.5, 0.1, 0.1])
dummy_noncrow_emb2 = np.array([-0.8, -0.7, 0.0, 0.2])

# Helper to calculate cosine similarity for numpy arrays (as used in simple_evaluate)
def cosine_similarity_numpy(arr1, arr2_batch):
    # arr1: 1D array (single embedding)
    # arr2_batch: 2D array (batch of embeddings)
    # Normalize arrays
    arr1_norm = arr1 / np.linalg.norm(arr1)
    arr2_batch_norm = arr2_batch / np.linalg.norm(arr2_batch, axis=1, keepdims=True)
    return np.dot(arr2_batch_norm, arr1_norm)


@patch('utilities.simple_evaluate.load_model_correct') # Mock model loading
@patch('utilities.simple_evaluate.load_sample_data')   # Mock data loading
def test_id_and_cvnc_metrics_calculated_correctly(mock_load_data, mock_load_model):
    """
    Tests if crow identification and crow vs non-crow metrics are calculated as expected
    based on a predefined set of embeddings and labels.
    """
    mock_load_model.return_value = MagicMock() # Dummy model, not used if embeddings are direct

    # Setup dummy data to be returned by load_sample_data
    test_embeddings_list = [
        dummy_crow_emb1_idA, dummy_crow_emb2_idA,
        dummy_crow_emb1_idB, dummy_crow_emb2_idB,
        dummy_noncrow_emb1, dummy_noncrow_emb2
    ]
    # Convert to torch tensors as load_sample_data is expected to return torch.stack(tensors)
    # However, evaluate_model converts them to numpy internally for cdist.
    # For this mock, let's assume load_sample_data returns the structure that evaluate_model then processes.
    # The current simple_evaluate.py's load_sample_data returns (torch.stack(images), labels)
    # and then evaluate_model computes embeddings.
    # To simplify, let's mock load_sample_data to return *already computed* embeddings (as numpy arrays)
    # and their string labels. This bypasses the actual model call within evaluate_model.

    # So, all_embeddings_np in evaluate_model will be this:
    mock_embeddings_np = np.array(test_embeddings_list)
    mock_labels_str = ['crowA', 'crowA', 'crowB', 'crowB', 'non-crow', 'non-crow']

    # We need to make sure load_sample_data returns something that leads to this state,
    # or that the part of evaluate_model that *generates* embeddings is also controlled.
    # The current evaluate_model takes image tensors from load_sample_data, then runs model.
    # Let's adjust: mock load_sample_data to return image tensors, and mock model(batch) call.

    dummy_image_tensors = torch.randn(len(mock_labels_str), 3, 512, 512) # Dummy image tensors
    mock_load_data.return_value = (dummy_image_tensors, mock_labels_str)

    # Mock the model's output when called with batches of these dummy tensors
    # The model in simple_evaluate is called batch by batch.
    # For simplicity, assume batch_size >= len(dummy_image_tensors) for this test.
    # Or, make the mock_model smarter.
    mock_model_instance = mock_load_model.return_value

    # We need to ensure the model's output is *our* mock_embeddings_np when it's called.
    # The model is called like: emb = model(batch)
    # And then results are appended to all_embeddings_np.
    # Let's make the mock model directly return slices of our mock_embeddings_np.
    def model_side_effect(batch_tensor):
        # This is a simplified mock. A real scenario might need to map input tensors to outputs.
        # Here, we assume it's called once with all tensors, or we need to track calls.
        # For this test, we'll assume it's called once with all dummy_image_tensors.
        # The evaluate_model batches, so this needs to be smarter or we simplify the test.

        # Simplification: Assume model is called ONCE and returns all embeddings.
        # This means we need to control the batching loop inside evaluate_model or make it call model once.
        # The current structure:
        # for i in range(0, len(all_image_tensors), batch_size):
        #    batch = all_image_tensors[i:i+batch_size].to(device)
        #    emb = model(batch) -> THIS IS WHAT WE MOCK
        #    all_embeddings_np.append(emb.cpu().numpy())
        # all_embeddings_np = np.vstack(all_embeddings_np)

        # Let's mock `model(batch)` to return corresponding parts of `mock_embeddings_np`
        # This requires knowing how `evaluate_model` batches.
        # Alternative: patch `np.vstack` or the loop itself.
        # Easiest: Assume batch_size in test context is large enough.
        # For this test, we will mock `model(batch)` to return the whole `mock_embeddings_np` if called once.
        # This is not ideal. Better to have the mock return parts.
        # Let's assume batch_size in evaluate_model is set to len(dummy_image_tensors) for test.
        # Or, the mock needs to be stateful.

        # If model is called with the whole batch of dummy_image_tensors:
        if batch_tensor.shape[0] == len(dummy_image_tensors):
             return torch.from_numpy(mock_embeddings_np).float() # Model returns torch tensor
        # If called in parts (e.g. batch_size=16 in evaluate_model)
        # This mock needs to be more complex. For now, let's assume test conditions make it simple.
        # A common pattern is to have side_effect return a list of values for successive calls.
        # For now, let's assume the test will ensure `model(batch)` is called in a way that we can return `mock_embeddings_np`
        # appropriately. The most direct way is to have it called once.
        # We will rely on the fact that the mock_model_instance is what's called.
        # We can set its return_value.
        # If batching is 16, and we have 6 samples, it's called once.
        mock_model_instance.return_value = torch.from_numpy(mock_embeddings_np).float()


    id_thresh = 0.85 # High threshold: emb1A and emb2A are similar, B1/B2 similar
                     # A and B are different.
    cvnc_thresh = 0.3 # Low threshold: non-crows should be very dissimilar from crows.

    results = evaluate_model(
        model_path="dummy_model.pth",
        base_dir="dummy_base_dir", # Not used if load_sample_data is mocked well
        device='cpu',
        id_similarity_threshold=id_thresh,
        non_crow_similarity_threshold=cvnc_thresh,
        max_crows=2, max_samples_per_crow=2 # Matches dummy data structure
    )

    # Expected ID metrics:
    # Pairs: (A1,A2), (A1,B1), (A1,B2), (A2,B1), (A2,B2), (B1,B2)
    # Sim(A1,A2) = cosine of [0.1,0.2,0.3,0.4] and [0.12,0.22,0.32,0.42] -> high (approx 0.99) -> TP
    # Sim(B1,B2) = cosine of [0.8,0.7,0.6,0.5] and [0.81,0.71,0.61,0.51] -> high (approx 0.99) -> TP
    # Sim(A1,B1) -> low (approx 0.36) -> TN
    # Sim(A1,B2) -> low (approx 0.35) -> TN
    # Sim(A2,B1) -> low (approx 0.37) -> TN
    # Sim(A2,B2) -> low (approx 0.36) -> TN
    # TP_id = 2 (A1-A2, B1-B2)
    # FP_id = 0 (no diff crows above id_thresh)
    # FN_id = 0 (no same crows below id_thresh)
    # TN_id = 4 (all diff crow pairs below id_thresh)
    assert results["tp_id"] == 2
    assert results["fp_id"] == 0
    assert results["fn_id"] == 0
    assert results["tn_id"] == 4
    assert results["precision_id"] == 1.0
    assert results["recall_id"] == 1.0
    assert results["f1_id"] == 1.0
    assert results["accuracy_id"] == 1.0

    # Expected CVNC metrics:
    # NonCrow1 vs (A1,A2,B1,B2). NonCrow2 vs (A1,A2,B1,B2)
    # Sim(NC1, A1) ~ -0.6. Sim(NC1, B1) ~ -0.9. Max sim for NC1 will be low. -> TN_nc
    # Sim(NC2, A1) ~ -0.5. Sim(NC2, B1) ~ -0.8. Max sim for NC2 will be low. -> TN_nc
    # So, TN_nc = 2, FP_nc = 0
    assert results["tn_nc"] == 2
    assert results["fp_nc"] == 0
    assert results["non_crow_true_rejection_rate"] == 1.0
    assert results["non_crow_false_alarm_rate"] == 0.0


@patch('utilities.simple_evaluate.load_model_correct')
@patch('utilities.simple_evaluate.load_sample_data')
def test_no_non_crow_samples_metrics(mock_load_data, mock_load_model):
    """Tests that CVNC metrics are handled gracefully if no non-crow samples are loaded."""
    mock_load_model.return_value = MagicMock()

    mock_embeddings_np = np.array([dummy_crow_emb1_idA, dummy_crow_emb2_idA, dummy_crow_emb1_idB])
    mock_labels_str = ['crowA', 'crowA', 'crowB'] # No "non-crow" labels

    dummy_image_tensors = torch.randn(len(mock_labels_str), 3, 512, 512)
    mock_load_data.return_value = (dummy_image_tensors, mock_labels_str)
    mock_load_model.return_value.return_value = torch.from_numpy(mock_embeddings_np).float()


    results = evaluate_model(
        model_path="dummy.pth", base_dir="dummy", device='cpu',
        id_similarity_threshold=0.5, non_crow_similarity_threshold=0.4
    )

    assert "non_crow_true_rejection_rate" not in results or results.get("non_crow_true_rejection_rate", 0) == 0
    assert "non_crow_false_alarm_rate" not in results or results.get("non_crow_false_alarm_rate", 0) == 0
    assert results.get("num_non_crow_samples_for_eval", 0) == 0
    # ID metrics should still be calculated
    assert "precision_id" in results
    assert results.get("num_crow_samples_for_eval", 0) == 3
