#!/usr/bin/env python3
"""
Simple Model Evaluation with Correct Architecture
Loads the model with 128-dimensional embeddings.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Added format
logger = logging.getLogger(__name__)
from collections import defaultdict # Added
from db import get_image_label # Added

class CrowResNet128(nn.Module):
    """ResNet model with 128-dimensional embeddings (matching your trained model)."""
    def __init__(self):
        super(CrowResNet128, self).__init__()
        import torchvision.models as models
        
        # Use ResNet18 backbone
        resnet = models.resnet18(weights=None)
        
        # Remove the last fully connected layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add custom fully connected layer for 128-dim embeddings
        self.fc = nn.Linear(512, 128)
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        
        # Get embeddings
        embeddings = self.fc(features)
        
        return embeddings

def load_model_correct(model_path, device):
    """Load model with correct architecture."""
    model = CrowResNet128().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_sample_data(base_dir, max_crows=20, max_samples_per_crow=10, num_non_crow_multiple=0.5, transform=None):
    """
    Loads a sample of data for evaluation from crop_metadata.json, including non-crow images.
    Filters images based on DB labels.
    """
    from PIL import Image # Moved import here as it's specific to this function now
    import torchvision.transforms as transforms # Moved import

    base_path = Path(base_dir)
    metadata_path = base_path / "metadata" / "crop_metadata.json"

    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}. Cannot load data.")
        return torch.empty(0), []
    
    try:
        with open(metadata_path, 'r') as f:
            crop_metadata = json.load(f)
    except Exception as e:
        logger.error(f"Error loading metadata file {metadata_path}: {e}")
        return torch.empty(0), []

    if "crops" not in crop_metadata or not crop_metadata["crops"]:
        logger.warning(f"No 'crops' key in metadata or no crops listed in {metadata_path}.")
        return torch.empty(0), []

    crow_images_map = defaultdict(list)
    non_crow_images_meta = [] # Stores (path, "non-crow")
    # Define labels to skip during evaluation loading - can be expanded
    labels_to_skip_eval = {'multi_crow', 'bad_crow', 'unknown_species', 'blurry', 'empty'}

    logger.info(f"Processing {len(crop_metadata['crops'])} items from metadata for data loading...")
    processed_paths = set() # To avoid processing same image path multiple times if metadata has duplicates

    for crop_relative_path_str, metadata_dict in crop_metadata["crops"].items():
        full_image_path = base_path / crop_relative_path_str

        if full_image_path in processed_paths:
            logger.debug(f"Skipping already processed path: {full_image_path}")
            continue
        processed_paths.add(full_image_path)

        if not full_image_path.exists():
            logger.debug(f"Image path {full_image_path} from metadata does not exist. Skipping.")
            continue

        label_info = get_image_label(str(full_image_path))
        current_label_from_db = label_info['label'] if label_info else None

        # This logic can be refined based on how strictly evaluation should adhere to DB labels vs metadata structure
        if current_label_from_db == 'not_a_crow':
            non_crow_images_meta.append((full_image_path, "non-crow"))
        elif current_label_from_db in labels_to_skip_eval:
            logger.debug(f"Skipping image {full_image_path} due to DB label: {current_label_from_db}")
            continue
        else: # Assumed to be a crow if not 'not_a_crow' or explicitly skipped
              # This includes 'crow' label, None (no DB label but in metadata), or other specific (non-skip) labels
            crow_id = metadata_dict.get('crow_id')
            if crow_id:
                # Optional: Further filter based on is_training_data if needed
                # if label_info and label_info.get('is_training_data', False):
                #     logger.debug(f"Skipping {full_image_path} (crow_id: {crow_id}) as it's training data.")
                #     continue
                crow_images_map[crow_id].append(full_image_path)
            else:
                logger.debug(f"Image {full_image_path} treated as crow-like but lacks crow_id in metadata. Skipping.")

    selected_samples_meta = []

    # Sample crow images
    crow_ids_available = list(crow_images_map.keys())
    if len(crow_ids_available) > max_crows:
        selected_crow_ids = random.sample(crow_ids_available, max_crows)
    else:
        selected_crow_ids = crow_ids_available
    
    logger.info(f"Selected {len(selected_crow_ids)} crow individuals for evaluation samples.")

    for crow_id in selected_crow_ids:
        images_for_crow = crow_images_map[crow_id]
        if len(images_for_crow) > max_samples_per_crow:
            selected_for_this_crow = random.sample(images_for_crow, max_samples_per_crow)
        else:
            selected_for_this_crow = images_for_crow
        for img_path in selected_for_this_crow:
            selected_samples_meta.append((img_path, crow_id))

    # Sample non-crow images
    num_max_crow_samples = len(selected_samples_meta) # Number of crow images actually selected
    num_non_crow_to_sample = int(num_max_crow_samples * num_non_crow_multiple) if num_max_crow_samples > 0 else int(max_crows * max_samples_per_crow * num_non_crow_multiple)
    # Fallback if no crow samples, base on potential max crow samples, or a fixed number like 50-100
    if num_non_crow_to_sample == 0 and num_non_crow_multiple > 0 : num_non_crow_to_sample = 50


    if not non_crow_images_meta:
        logger.info("No 'not_a_crow' images found or loaded from metadata.")
    elif num_non_crow_to_sample > 0:
        if len(non_crow_images_meta) > num_non_crow_to_sample:
            selected_samples_meta.extend(random.sample(non_crow_images_meta, num_non_crow_to_sample))
        else:
            selected_samples_meta.extend(non_crow_images_meta) # Add all available non-crows
    logger.info(f"Selected {len(selected_samples_meta) - num_max_crow_samples} non-crow images.")
    logger.info(f"Total samples selected for loading (crows + non-crows): {len(selected_samples_meta)}")

    all_image_tensors = []
    all_labels_str = []

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((512, 512)), # Match model input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    for img_path, label_str in selected_samples_meta:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            all_image_tensors.append(img_tensor)
            all_labels_str.append(label_str)
        except Exception as e:
            logger.error(f"Error loading or transforming image {img_path}: {e}")
            continue
    
    if not all_image_tensors:
        logger.warning("No images were successfully loaded into tensors for evaluation.")
        return torch.empty(0), []

    return torch.stack(all_image_tensors), all_labels_str


def evaluate_model(model_path, base_dir, device=None, # Changed crop_dir to base_dir
                   id_similarity_threshold=0.5, non_crow_similarity_threshold=0.4, # Added new thresholds
                   max_crows=20, max_samples_per_crow=10): # Added sampling params here
    """Evaluate the model on sample data, including crow vs non-crow distinction."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = load_model_correct(model_path, device)
    logger.info("Model loaded successfully")
    
    # Load data using the updated function, passing necessary sampling parameters
    all_image_tensors, all_labels_str = load_sample_data(
        base_dir,
        max_crows=max_crows,
        max_samples_per_crow=max_samples_per_crow
    )

    if all_image_tensors.numel() == 0:
        logger.error("No data loaded. Aborting evaluation.")
        return None
    logger.info(f"Loaded {len(all_labels_str)} total samples for evaluation from metadata.")

    # Compute embeddings for all loaded images
    all_embeddings_np = []
    batch_size = 16 # Or make this a parameter
    with torch.no_grad():
        for i in range(0, len(all_image_tensors), batch_size):
            batch = all_image_tensors[i:i+batch_size].to(device)
            emb = model(batch)
            all_embeddings_np.append(emb.cpu().numpy())
    
    if not all_embeddings_np:
        logger.error("Failed to compute any embeddings.")
        return None
    all_embeddings_np = np.vstack(all_embeddings_np)
    logger.info(f"Computed {all_embeddings_np.shape[0]} embeddings of dimension {all_embeddings_np.shape[1]}.")

    # Separate crow and non-crow data
    crow_embeddings_list = []
    crow_labels_actual_ids = []
    non_crow_embeddings_list = []
    
    for i, label_str in enumerate(all_labels_str):
        if label_str == "non-crow":
            non_crow_embeddings_list.append(all_embeddings_np[i])
        else: # It's a crow_id
            crow_embeddings_list.append(all_embeddings_np[i])
            crow_labels_actual_ids.append(label_str)

    logger.info(f"Separated data: {len(crow_labels_actual_ids)} crow samples from {len(set(crow_labels_actual_ids))} unique individuals, "
                f"{len(non_crow_embeddings_list)} non-crow samples.")

    metrics = {
        "id_similarity_threshold": id_similarity_threshold,
        "non_crow_similarity_threshold": non_crow_similarity_threshold,
        "num_total_samples_loaded": len(all_labels_str),
        "num_crow_samples_for_eval": len(crow_labels_actual_ids),
        "num_unique_crows_for_eval": len(set(crow_labels_actual_ids)),
        "num_non_crow_samples_for_eval": len(non_crow_embeddings_list)
    }

    # 1. Crow Identification Metrics
    if crow_embeddings_list:
        crow_embeddings_np = np.array(crow_embeddings_list) # Convert list of arrays to 2D numpy array

        # Pairwise cosine distances for crow embeddings
        distances_id = cdist(crow_embeddings_np, crow_embeddings_np, metric='cosine')
        similarities_id = 1 - distances_id

        gt_matrix_id = np.array([[crow_labels_actual_ids[i] == crow_labels_actual_ids[j]
                                 for j in range(len(crow_labels_actual_ids))]
                                 for i in range(len(crow_labels_actual_ids))])

        mask_id = ~np.eye(len(crow_labels_actual_ids), dtype=bool)
        similarities_flat_id = similarities_id[mask_id]
        gt_flat_id = gt_matrix_id[mask_id]

        predictions_id = similarities_flat_id >= id_similarity_threshold

        tp_id = np.sum(predictions_id & gt_flat_id)
        fp_id = np.sum(predictions_id & ~gt_flat_id)
        fn_id = np.sum(~predictions_id & gt_flat_id)
        tn_id = np.sum(~predictions_id & ~gt_flat_id) # True negatives for ID are correctly identified different crow pairs

        metrics["precision_id"] = tp_id / (tp_id + fp_id) if (tp_id + fp_id) > 0 else 0
        metrics["recall_id"] = tp_id / (tp_id + fn_id) if (tp_id + fn_id) > 0 else 0
        metrics["f1_id"] = 2 * (metrics["precision_id"] * metrics["recall_id"]) / \
                           (metrics["precision_id"] + metrics["recall_id"]) if (metrics["precision_id"] + metrics["recall_id"]) > 0 else 0
        metrics["accuracy_id"] = (tp_id + tn_id) / (tp_id + fp_id + fn_id + tn_id) if (tp_id + fp_id + fn_id + tn_id) > 0 else 0

        same_crow_sims_id = similarities_flat_id[gt_flat_id]
        diff_crow_sims_id = similarities_flat_id[~gt_flat_id]
        metrics["avg_similarity_pos_id"] = np.mean(same_crow_sims_id) if len(same_crow_sims_id) > 0 else 0
        metrics["avg_similarity_neg_id"] = np.mean(diff_crow_sims_id) if len(diff_crow_sims_id) > 0 else 0
        metrics["separability_id"] = metrics["avg_similarity_pos_id"] - metrics["avg_similarity_neg_id"]
        metrics.update({"tp_id": int(tp_id), "fp_id": int(fp_id), "fn_id": int(fn_id), "tn_id": int(tn_id)})
    else:
        logger.warning("No crow samples available for Crow Identification metrics.")

    # 2. Crow vs. Non-Crow Distinction Metrics
    if non_crow_embeddings_list and crow_embeddings_list:
        non_crow_embeddings_np = np.array(non_crow_embeddings_list)
        # crow_embeddings_np already exists from ID metrics section

        true_negatives_nc = 0  # Non-crow correctly rejected
        false_positives_nc = 0 # Non-crow mistaken for a known crow

        for i in range(non_crow_embeddings_np.shape[0]):
            nc_emb_single = non_crow_embeddings_np[i, :].reshape(1, -1)
            # Similarities of this non-crow embedding to all known crow embeddings
            sims_to_crows = 1 - cdist(nc_emb_single, crow_embeddings_np, metric='cosine')
            max_sim_to_any_crow = np.max(sims_to_crows) if sims_to_crows.size > 0 else -1 # handle case of no crow embeddings (though checked by if)

            if max_sim_to_any_crow < non_crow_similarity_threshold:
                true_negatives_nc += 1
            else:
                false_positives_nc += 1

        total_actual_non_crows = len(non_crow_embeddings_list)
        metrics["non_crow_true_rejection_rate"] = true_negatives_nc / total_actual_non_crows if total_actual_non_crows > 0 else 0
        metrics["non_crow_false_alarm_rate"] = false_positives_nc / total_actual_non_crows if total_actual_non_crows > 0 else 0
        metrics.update({"tn_nc": true_negatives_nc, "fp_nc": false_positives_nc})
    else:
        logger.warning("Not enough data for Crow vs. Non-Crow distinction (need both crow and non-crow samples).")

    # Print results
    print("\n" + "="*60)
    print("FACEBEAK MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Dataset: {metrics.get('num_total_samples_loaded',0)} total samples loaded.")
    print(f"  Crow samples for ID eval: {metrics.get('num_crow_samples_for_eval',0)} from {metrics.get('num_unique_crows_for_eval',0)} unique crows.")
    print(f"  Non-crow samples for CvNC eval: {metrics.get('num_non_crow_samples_for_eval',0)}.")
    print(f"Embedding dimension: {all_embeddings_np.shape[1] if all_embeddings_np.size > 0 else 'N/A'}")
    print("\nCrow Identification Metrics:")
    print(f"  Threshold: {metrics.get('id_similarity_threshold', 'N/A'):.2f}")
    print(f"  Precision (ID): {metrics.get('precision_id', 0):.3f}")
    print(f"  Recall (ID):    {metrics.get('recall_id', 0):.3f}")
    print(f"  F1 Score (ID):  {metrics.get('f1_id', 0):.3f}")
    print(f"  Accuracy (Pair ID): {metrics.get('accuracy_id', 0):.3f}")
    print(f"  Same Crow Sim (ID):    {metrics.get('avg_similarity_pos_id',0):.3f} ± {np.std(same_crow_sims_id) if 'avg_similarity_pos_id' in metrics and len(same_crow_sims_id)>0 else 0:.3f}")
    print(f"  Different Crow Sim (ID): {metrics.get('avg_similarity_neg_id',0):.3f} ± {np.std(diff_crow_sims_id) if 'avg_similarity_neg_id' in metrics and len(diff_crow_sims_id)>0 else 0:.3f}")
    print(f"  Separability (ID):      {metrics.get('separability_id',0):.3f}")

    print("\nCrow vs. Non-Crow Distinction Metrics:")
    print(f"  Threshold: {metrics.get('non_crow_similarity_threshold', 'N/A'):.2f}")
    print(f"  Non-Crow True Rejection Rate: {metrics.get('non_crow_true_rejection_rate', 0):.3f} (Non-crows correctly rejected as not any known crow)")
    print(f"  Non-Crow False Alarm Rate:    {metrics.get('non_crow_false_alarm_rate', 0):.3f} (Non-crows mistaken for a known crow)")
    print("="*60)
    
    return metrics

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple model evaluation with crow vs non-crow distinction.')
    parser.add_argument('--model-path', default='crow_resnet_triplet.pth', help='Path to model')
    parser.add_argument('--base-dir', default='crow_crops_eval', # Changed from crop-dir
                        help='Base directory with evaluation image crops and metadata.json.')
    parser.add_argument('--id-similarity-threshold', type=float, default=0.5,
                        help="Threshold for crow identification metrics.")
    parser.add_argument('--non-crow-similarity-threshold', type=float, default=0.4,
                        help="Threshold for distinguishing non-crows from known crows.")
    parser.add_argument('--max-crows', type=int, default=20, help="Max crow individuals to sample.")
    parser.add_argument('--max-samples-per-crow', type=int, default=10, help="Max samples per crow.")
    # Add device argument if not already present or handled by load_model_correct
    parser.add_argument('--device', type=str, default=None, help="Device: 'cuda' or 'cpu'. Auto-detects if None.")


    args = parser.parse_args()
    
    try:
        results = evaluate_model(
            args.model_path,
            args.base_dir,
            device=args.device,
            id_similarity_threshold=args.id_similarity_threshold,
            non_crow_similarity_threshold=args.non_crow_similarity_threshold,
            max_crows=args.max_crows,
            max_samples_per_crow=args.max_samples_per_crow
        )
        print(f"\nEvaluation complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 