import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import cv2
import os
from datetime import datetime
import logging
from db import get_crow_embeddings, get_all_crows
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics.pairwise import pairwise_distances
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crow_clustering.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Helper to convert all bools to int for JSON serialization
def convert_bools(obj):
    if isinstance(obj, dict):
        return {k: convert_bools(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_bools(v) for v in obj]
    elif isinstance(obj, bool):
        return int(obj)
    else:
        return obj

class CrowClusterAnalyzer:
    def __init__(self, eps_range=(0.2, 0.5), min_samples_range=(2, 5), similarity_threshold=0.7, temporal_weight=0.1):
        """
        Initialize the crow cluster analyzer.
        
        Args:
            eps_range: Tuple of (min, max) values to try for DBSCAN eps parameter
            min_samples_range: Tuple of (min, max) values to try for DBSCAN min_samples parameter
            similarity_threshold: Minimum average similarity for a valid cluster
            temporal_weight: Weight for temporal proximity in distance calculation
        """
        self.eps_range = eps_range
        self.min_samples_range = min_samples_range
        self.similarity_threshold = similarity_threshold
        self.temporal_weight = temporal_weight
        self.best_params = None
        self.cluster_metrics = {}
        
    def extract_embeddings(self, crow_id: int) -> np.ndarray:
        """Extract all embeddings for a given crow."""
        embeddings = get_crow_embeddings(crow_id)
        if not embeddings:
            return np.array([])
        return np.array([e['embedding'] for e in embeddings])
    
    def validate_clusters(self, labels: np.ndarray, embeddings: np.ndarray) -> Tuple[List[int], Dict]:
        """
        Validate clusters by checking internal similarity and other metrics.
        
        Args:
            labels: Cluster labels from DBSCAN
            embeddings: Crow embeddings
            
        Returns:
            Tuple of (valid_cluster_ids, validation_metrics)
        """
        if len(labels) == 0 or len(embeddings) == 0:
            return [], {
                'total_clusters': 0,
                'valid_clusters': 0,
                'invalid_clusters': 0,
                'valid_cluster_ratio': 0.0,
                'cluster_details': {}
            }
        
        valid_clusters = []
        validation_metrics = {
            'total_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'valid_clusters': 0,
            'invalid_clusters': 0,
            'cluster_details': {}
        }
        
        for cluster_id in set(labels):
            if cluster_id == -1:  # Skip noise
                continue
                
            # Get embeddings for this cluster
            cluster_mask = labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            
            if len(cluster_embeddings) < 2:  # Skip clusters with only one point
                validation_metrics['invalid_clusters'] += 1
                continue
            
            # Calculate pairwise similarities
            similarities = np.dot(cluster_embeddings, cluster_embeddings.T)
            np.fill_diagonal(similarities, 0)  # Exclude self-similarity
            
            # Calculate metrics
            mean_similarity = float(np.mean(similarities))  # Ensure float type
            std_similarity = float(np.std(similarities))    # Ensure float type
            cluster_size = len(cluster_embeddings)
            
            is_valid = mean_similarity > self.similarity_threshold
            
            cluster_metrics = {
                'size': cluster_size,
                'mean_similarity': mean_similarity,
                'std_similarity': std_similarity,
                'is_valid': bool(is_valid)  # Ensure boolean type for test compatibility
            }
            
            validation_metrics['cluster_details'][f'cluster_{cluster_id}'] = cluster_metrics
            
            if is_valid:
                valid_clusters.append(cluster_id)
                validation_metrics['valid_clusters'] += 1
            else:
                validation_metrics['invalid_clusters'] += 1
                logger.warning(
                    f"Cluster {cluster_id} has low internal similarity: {mean_similarity:.3f} "
                    f"(threshold: {self.similarity_threshold})"
                )
        
        # Add summary metrics
        validation_metrics['valid_cluster_ratio'] = float(
            validation_metrics['valid_clusters'] / validation_metrics['total_clusters']
            if validation_metrics['total_clusters'] > 0 else 0.0
        )
        
        return valid_clusters, validation_metrics

    def find_optimal_params(self, embeddings: np.ndarray, frame_numbers: Optional[List[int]] = None, confidences: Optional[List[float]] = None, output_dir: Optional[str] = None) -> Tuple[float, int]:
        """
        Find optimal DBSCAN parameters using grid search with validation.
        
        Args:
            embeddings: Array of crow embeddings
            frame_numbers: List of frame numbers for temporal consistency
            confidences: List of confidence scores to weight embeddings
            output_dir: Output directory for saving visualization
            
        Returns:
            Tuple of (best_eps, best_min_samples)
        """
        if len(embeddings) < 2:
            raise ValueError("Need at least 2 embeddings for clustering")
        
        if frame_numbers is not None and len(frame_numbers) != len(embeddings):
            raise ValueError("Number of frame numbers must match number of embeddings")
        
        if confidences is not None:
            confidences = np.array(confidences, dtype=np.float64)
            if np.any(confidences < 0) or np.any(confidences > 1):
                raise ValueError("Invalid confidence scores")
        
        best_score = -1
        best_params = None
        results = []
        
        # Adjust parameter ranges based on data size
        n_samples = len(embeddings)
        min_samples_range = (
            min(self.min_samples_range[0], max(2, n_samples // 10)),
            min(self.min_samples_range[1], max(3, n_samples // 5))
        )
        
        # Grid search over parameter space
        eps_values = np.linspace(self.eps_range[0], self.eps_range[1], min(10, n_samples))
        min_samples_values = range(min_samples_range[0], min_samples_range[1] + 1)
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    # Skip if min_samples is too large for the dataset
                    if min_samples >= n_samples:
                        continue
                    
                    # Run DBSCAN
                    if frame_numbers is not None:
                        dist_matrix = self.compute_distance_matrix_with_temporal(embeddings, frame_numbers, confidences)
                        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(dist_matrix)
                    else:
                        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
                    labels = clustering.labels_
                    
                    # Skip if all points are noise or only one cluster
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters <= 1:
                        continue
                    
                    # Skip if too many noise points
                    n_noise = list(labels).count(-1)
                    if n_noise > n_samples * 0.5:  # More than 50% noise
                        continue
                    
                    # Validate clusters
                    valid_clusters, validation_metrics = self.validate_clusters(labels, embeddings)
                    
                    # Skip if no valid clusters
                    if not valid_clusters:
                        continue
                    
                    # Calculate metrics
                    try:
                        # Silhouette score (higher is better)
                        sil_score = silhouette_score(embeddings, labels)
                        # Calinski-Harabasz score (higher is better)
                        ch_score = calinski_harabasz_score(embeddings, labels)
                    except Exception as e:
                        logger.warning(f"Error calculating metrics: {e}")
                        continue
                    
                    # Combined score with validation metrics
                    combined_score = (
                        0.4 * sil_score +  # Clustering quality
                        0.3 * ch_score +   # Cluster separation
                        0.3 * validation_metrics['valid_cluster_ratio']  # Validation quality
                    )
                    
                    results.append({
                        'eps': float(eps),
                        'min_samples': int(min_samples),
                        'n_clusters': int(n_clusters),
                        'n_noise': int(n_noise),
                        'silhouette_score': float(sil_score),
                        'calinski_harabasz_score': float(ch_score),
                        'valid_cluster_ratio': float(validation_metrics['valid_cluster_ratio']),
                        'combined_score': float(combined_score),
                        'validation_metrics': validation_metrics
                    })
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_params = (float(eps), int(min_samples))
                        
                except Exception as e:
                    logger.warning(f"Error with params eps={eps}, min_samples={min_samples}: {str(e)}")
                    continue
        
        if not results:
            # If no valid parameters found, use conservative defaults
            logger.warning("No valid clustering parameters found, using conservative defaults")
            best_params = (self.eps_range[0], min_samples_range[0])
            best_score = 0.0
            # Still save empty results for consistency
            self.cluster_metrics['parameter_search'] = []
        else:
            # Save parameter search results
            self.cluster_metrics['parameter_search'] = convert_bools(results)
        
        self.best_params = best_params
        
        # Log best parameters and metrics only if we found valid results
        if results:
            best_result = next(r for r in results if r['eps'] == best_params[0] and r['min_samples'] == best_params[1])
            logger.info("Best clustering parameters found:")
            logger.info(f"eps={best_params[0]:.3f}, min_samples={best_params[1]}")
            logger.info(f"Metrics:")
            logger.info(f"- Combined score: {best_score:.3f}")
            logger.info(f"- Silhouette score: {best_result['silhouette_score']:.3f}")
            logger.info(f"- Valid cluster ratio: {best_result['valid_cluster_ratio']:.3f}")
            logger.info(f"- Number of clusters: {best_result['n_clusters']}")
            logger.info(f"- Number of noise points: {best_result['n_noise']}")
            
            # Plot parameter search results
            self._plot_parameter_search(results, output_dir)
        
        return best_params
    
    def _plot_parameter_search(self, results: List[Dict], output_dir: Optional[str] = None):
        """Plot parameter search results."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pathlib import Path
        plt.figure(figsize=(12, 8))
        eps_values = sorted(set(r['eps'] for r in results))
        min_samples_values = sorted(set(r['min_samples'] for r in results))
        score_matrix = np.zeros((len(eps_values), len(min_samples_values)))
        for r in results:
            i = eps_values.index(r['eps'])
            j = min_samples_values.index(r['min_samples'])
            score_matrix[i, j] = r['combined_score']
        plt.subplot(2, 2, 1)
        sns.heatmap(score_matrix, xticklabels=min_samples_values, yticklabels=eps_values,
                   cmap='viridis', annot=True, fmt='.2f')
        plt.title('Combined Score')
        plt.xlabel('Min Samples')
        plt.ylabel('Eps')
        cluster_matrix = np.zeros((len(eps_values), len(min_samples_values)))
        for r in results:
            i = eps_values.index(r['eps'])
            j = min_samples_values.index(r['min_samples'])
            cluster_matrix[i, j] = r['n_clusters']
        plt.subplot(2, 2, 2)
        sns.heatmap(cluster_matrix, xticklabels=min_samples_values, yticklabels=eps_values,
                   cmap='viridis', annot=True, fmt='.0f')
        plt.title('Number of Clusters')
        plt.xlabel('Min Samples')
        plt.ylabel('Eps')
        plt.tight_layout()
        if output_dir is not None:
            output_path = Path(output_dir) / 'clustering_parameter_search.png'
        else:
            output_path = Path('clustering_parameter_search.png')
        try:
            plt.savefig(str(output_path), format='png', dpi=100, bbox_inches='tight')
        except (TypeError, AttributeError) as e:
            # Fallback for PIL/matplotlib compatibility issues
            logger.warning(f"Failed to save plot with PIL backend: {e}")
            try:
                # Try without format specification
                plt.savefig(str(output_path), dpi=100, bbox_inches='tight')
            except Exception as e2:
                logger.warning(f"Failed to save plot even without format: {e2}")
                # Skip saving the plot if all methods fail
        plt.close()
    
    def cluster_crows(self, embeddings: np.ndarray, frame_numbers: Optional[List[int]] = None, 
                     eps: Optional[float] = None, min_samples: Optional[int] = None, confidences: Optional[List[float]] = None, output_dir: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
        """
        Cluster crow embeddings using DBSCAN with validation.
        
        Args:
            embeddings: Array of crow embeddings
            frame_numbers: List of frame numbers for temporal consistency
            eps: DBSCAN eps parameter (if None, will use optimal value)
            min_samples: DBSCAN min_samples parameter (if None, will use optimal value)
            confidences: List of confidence scores to weight embeddings
            output_dir: Output directory for saving visualization
            
        Returns:
            Tuple of (cluster labels, metrics dictionary)
        """
        # Validate confidence scores first
        if confidences is not None:
            confidences = np.array(confidences, dtype=np.float64)
            if np.any(confidences < 0) or np.any(confidences > 1):
                raise ValueError("Invalid confidence scores")
        # Validate input parameters
        if len(embeddings) < 2:
            raise ValueError("Need at least 2 embeddings for clustering")
        if frame_numbers is not None and len(frame_numbers) != len(embeddings):
            raise ValueError("Number of frame numbers must match number of embeddings")
        # Validate clustering parameters
        if eps is not None:
            if not isinstance(eps, (int, float)) or eps <= 0:
                raise ValueError("Invalid eps parameter: must be a positive number")
        if min_samples is not None:
            if not isinstance(min_samples, int) or min_samples < 1:
                raise ValueError("Invalid min_samples parameter: must be a positive integer")
        # Get optimal parameters if not provided
        if eps is None or min_samples is None:
            if self.best_params is None:
                eps, min_samples = self.find_optimal_params(embeddings, frame_numbers, confidences, output_dir)
            else:
                eps, min_samples = self.best_params
        # Run DBSCAN
        try:
            if frame_numbers is not None:
                dist_matrix = self.compute_distance_matrix_with_temporal(embeddings, frame_numbers, confidences)
                clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(dist_matrix)
            else:
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
        except Exception as e:
            if "eps" in str(e):
                raise ValueError("Invalid eps parameter") from e
            elif "min_samples" in str(e):
                raise ValueError("Invalid min_samples parameter") from e
            else:
                raise
        labels = clustering.labels_
        valid_clusters, validation_metrics = self.validate_clusters(labels, embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        metrics = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'eps': eps,
            'min_samples': min_samples,
            'validation_metrics': validation_metrics
        }
        if n_clusters > 1:
            metrics.update({
                'silhouette_score': silhouette_score(embeddings, labels),
                'calinski_harabasz_score': calinski_harabasz_score(embeddings, labels)
            })
        logger.info("Clustering results:")
        logger.info(f"Total clusters: {n_clusters}")
        logger.info(f"Valid clusters: {validation_metrics['valid_clusters']}")
        logger.info(f"Invalid clusters: {validation_metrics['invalid_clusters']}")
        logger.info(f"Valid cluster ratio: {validation_metrics['valid_cluster_ratio']:.3f}")
        logger.info(f"Number of noise points: {n_noise}")
        self._visualize_clusters(embeddings, labels, output_dir)
        self.cluster_metrics['final_clustering'] = metrics
        def extract_primitives(obj):
            if isinstance(obj, dict):
                return {k: extract_primitives(v) for k, v in obj.items() if isinstance(k, (str, int, float))}
            elif isinstance(obj, list):
                return [extract_primitives(v) for v in obj]
            elif isinstance(obj, (int, float, str)):
                return obj
            else:
                return str(obj)
        results_to_save = {}
        if 'parameter_search' in self.cluster_metrics and self.cluster_metrics['parameter_search']:
            best_result = self.cluster_metrics['parameter_search'][-1]
            results_to_save['best_parameter_search'] = extract_primitives(best_result)
        if 'final_clustering' in self.cluster_metrics:
            final_clustering = self.cluster_metrics['final_clustering']
            results_to_save['final_clustering'] = extract_primitives(final_clustering)
        with open('clustering_metrics.json', 'w') as f:
            json.dump(results_to_save, f, indent=2)
        return labels, metrics
    
    def _visualize_clusters(self, embeddings: np.ndarray, labels: np.ndarray, output_dir: Optional[str] = None):
        """Visualize clusters using t-SNE."""
        if len(embeddings) < 5:
            logger.warning("Skipping cluster visualization: too few samples")
            return
        try:
            from pathlib import Path
            perplexity = min(30, len(embeddings) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            embeddings_2d = tsne.fit_transform(embeddings)
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Cluster')
            plt.title('t-SNE visualization of crow clusters')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            if -1 in labels:
                plt.scatter([], [], c='gray', label='Noise')
                plt.legend()
            # Save plot in output_dir if provided
            if output_dir is not None:
                output_path = Path(output_dir) / 'crow_clusters_visualization.png'
            else:
                output_path = Path('crow_clusters_visualization.png')
            try:
                plt.savefig(str(output_path), format='png', dpi=100, bbox_inches='tight')
            except (TypeError, AttributeError) as e:
                # Fallback for PIL/matplotlib compatibility issues
                logger.warning(f"Failed to save plot with PIL backend: {e}")
                try:
                    # Try without format specification
                    plt.savefig(str(output_path), dpi=100, bbox_inches='tight')
                except Exception as e2:
                    logger.warning(f"Failed to save plot even without format: {e2}")
                    # Skip saving the plot if all methods fail
            plt.close()
            if not output_path.exists():
                logger.warning("Failed to save cluster visualization: file not created")
        except Exception as e:
            logger.warning(f"Failed to visualize clusters: {e}")
            # Don't re-raise the exception, just log it

    def compute_distance_matrix_with_temporal(self, embeddings, frame_numbers, confidences=None):
        """
        Compute distance matrix with temporal weighting.
        
        Args:
            embeddings: numpy array of shape (n_samples, n_features)
            frame_numbers: list of frame numbers for temporal consistency
            confidences: list of confidence scores to weight embeddings
        
        Returns:
            numpy array: distance matrix with temporal weighting
        """
        if len(embeddings) < 2:
            raise ValueError("Need at least 2 embeddings for distance computation")
        
        if len(frame_numbers) != len(embeddings):
            raise ValueError("Number of frame numbers must match number of embeddings")
        
        if confidences is not None and len(confidences) != len(embeddings):
            raise ValueError("Number of confidence scores must match number of embeddings")
        
        n = len(embeddings)
        
        # Compute cosine distance matrix for embeddings
        dist_matrix = pairwise_distances(embeddings, metric='cosine')
        
        # Normalize frame numbers for temporal distance
        frame_numbers = np.array(frame_numbers, dtype=np.float64)
        
        # Handle frame number gaps by using log scale
        frame_diff = np.abs(frame_numbers[:, None] - frame_numbers[None, :])
        # Add small epsilon to avoid log(0)
        frame_diff = np.log1p(frame_diff)
        # Normalize to [0, 1] range
        max_frame_diff = np.max(frame_diff) if np.max(frame_diff) > 0 else 1
        temporal_dist = frame_diff / max_frame_diff
        
        # Adjust distance matrix based on confidences if provided
        if confidences is not None:
            confidences = np.array(confidences, dtype=np.float64)
            if np.any(confidences < 0) or np.any(confidences > 1):
                raise ValueError("Confidence scores must be between 0 and 1")
            
            # Normalize confidences to range [0.5, 1.5] to avoid extreme scaling
            # This ensures that even low confidence points can still form clusters
            weights = 0.5 + confidences
            # Scale distances inversely with confidence (higher confidence = lower distance)
            weight_matrix = np.outer(weights, weights)
            dist_matrix = dist_matrix * (2 - weight_matrix)
        
        # Combine distances with temporal weighting
        # Use a sigmoid function to smoothly blend the two distances
        alpha = self.temporal_weight
        combined_dist = (1 - alpha) * dist_matrix + alpha * temporal_dist
        
        # Ensure all distances are in [0, 1] range
        max_dist = np.max(combined_dist)
        if max_dist > 0:
            combined_dist = combined_dist / max_dist
        
        return combined_dist

def process_video_for_clustering(video_path: str, output_dir: str, 
                               eps: Optional[float] = None,
                               min_samples: Optional[int] = None) -> Dict:
    """Process a video to cluster crow embeddings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analyzer = CrowClusterAnalyzer()
    crows = get_all_crows()
    all_embeddings = []
    crow_ids = []
    frame_numbers = []
    confidences = []
    logger.info("Extracting embeddings for all crows...")
    for crow in tqdm(crows):
        embeddings = crow.get('embeddings', [])
        if len(embeddings) > 0:
            all_embeddings.extend(embeddings)
            crow_ids.extend([crow['id']] * len(embeddings))
            frame_numbers.extend([crow['frame_number']] * len(embeddings))
            confidences.extend([crow.get('confidence', 0.5)] * len(embeddings))
    if not all_embeddings:
        logger.warning("No embeddings found for clustering")
        return {}
    all_embeddings = np.array(all_embeddings)
    crow_ids = np.array(crow_ids)
    frame_numbers = np.array(frame_numbers)
    confidences = np.array(confidences)
    # Always call find_optimal_params to generate the parameter search plot
    analyzer.find_optimal_params(all_embeddings, frame_numbers, confidences, str(output_dir))
    if eps is None or min_samples is None:
        logger.info("Finding optimal clustering parameters...")
        eps, min_samples = analyzer.best_params
    logger.info(f"Clustering with eps={eps:.3f}, min_samples={min_samples}")
    labels, metrics = analyzer.cluster_crows(all_embeddings, frame_numbers, eps, min_samples, confidences, str(output_dir))
    
    # Save results
    results = {
        'parameters': {'eps': eps, 'min_samples': min_samples},
        'metrics': metrics,
        'clusters': {}
    }
    
    # Group embeddings by cluster
    unique_labels = set(labels) - {-1}  # Exclude noise
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_crow_ids = crow_ids[cluster_mask].tolist()
        results['clusters'][f'cluster_{label}'] = cluster_crow_ids
    
    # Save noise points
    noise_mask = labels == -1
    if np.any(noise_mask):
        results['noise'] = crow_ids[noise_mask].tolist()
    
    # Save to file
    output_file = output_dir / "clustering_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Clustering results saved to {output_file}")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cluster crow embeddings from video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output-dir", default="clustering_results", help="Output directory")
    parser.add_argument("--eps-min", type=float, default=0.2, help="Minimum DBSCAN eps parameter")
    parser.add_argument("--eps-max", type=float, default=0.5, help="Maximum DBSCAN eps parameter")
    parser.add_argument("--min-samples-min", type=int, default=2, help="Minimum DBSCAN min_samples parameter")
    parser.add_argument("--min-samples-max", type=int, default=5, help="Maximum DBSCAN min_samples parameter")
    args = parser.parse_args()
    
    # Initialize analyzer with custom parameter ranges
    analyzer = CrowClusterAnalyzer(
        eps_range=(args.eps_min, args.eps_max),
        min_samples_range=(args.min_samples_min, args.min_samples_max)
    )
    
    process_video_for_clustering(args.video, args.output_dir) 