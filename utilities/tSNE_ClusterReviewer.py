# tSNE_ClusterReviewer.py
# This script visualizes the embedding space of your trained crow model using t-SNE,
# optionally overlays image thumbnails, and flags clusters or outliers for manual review.

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from collections import defaultdict, Counter
from pathlib import Path
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import get_all_crows, get_crow_embeddings

# --- Config ---
EMBEDDING_DIM = 512
OUTPUT_DIR = "tsne_output"
PERPLEXITY = 30
IMAGE_OVERLAY = True  # Set to False for headless servers
MAX_IMAGES = 1000
MIN_SAMPLES_PER_CROW = 3  # Only analyze crows with enough samples

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_embeddings_with_metadata():
    """Load embeddings with comprehensive metadata."""
    print("🔍 Loading embeddings from database...")
    
    all_embeddings = []
    labels = []
    crow_names = []
    video_paths = []
    frame_numbers = []
    confidences = []
    
    crows = get_all_crows()
    if not crows:
        print("❌ No crows found in database. Run video processing first!")
        return None
    
    print(f"Found {len(crows)} crows in database")
    
    valid_crows = 0
    for crow in crows:
        crow_id = crow['id']
        crow_name = crow.get('name', f'Crow_{crow_id}')
        entries = get_crow_embeddings(crow_id)
        
        if len(entries) < MIN_SAMPLES_PER_CROW:
            print(f"⚠️ Skipping Crow {crow_id}: only {len(entries)} samples (need {MIN_SAMPLES_PER_CROW}+)")
            continue
            
        valid_crows += 1
        print(f"✅ Crow {crow_id} ({crow_name}): {len(entries)} embeddings")
        
        for emb in entries:
            all_embeddings.append(emb['embedding'])
            labels.append(str(crow_id))
            crow_names.append(crow_name)
            video_paths.append(emb.get('video_path', 'unknown'))
            frame_numbers.append(emb.get('frame_number', 0))
            confidences.append(emb.get('confidence', 1.0))
    
    if not all_embeddings:
        print("❌ No valid embeddings found. Need more training data!")
        return None
        
    print(f"📊 Loaded {len(all_embeddings)} embeddings from {valid_crows} crows")
    return {
        'embeddings': np.vstack(all_embeddings),
        'labels': labels,
        'crow_names': crow_names, 
        'video_paths': video_paths,
        'frame_numbers': frame_numbers,
        'confidences': confidences
    }

def run_tsne_analysis(data):
    """Run t-SNE with multiple perplexity values."""
    print("🧠 Running t-SNE analysis...")
    
    X = data['embeddings']
    X_scaled = StandardScaler().fit_transform(X)
    
    results = {}
    perplexities = [5, 15, 30, 50] if len(X) > 100 else [5, 15]
    
    for perp in perplexities:
        if perp >= len(X):
            continue
            
        print(f"  Running t-SNE with perplexity={perp}...")
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
        X_embedded = tsne.fit_transform(X_scaled)
        results[f'perp_{perp}'] = X_embedded
    
    return results, X_scaled

def analyze_clusters(X_scaled, labels):
    """Analyze natural clusters in the data."""
    print("🔬 Analyzing clusters...")
    
    # Try different DBSCAN parameters
    eps_values = [0.5, 1.0, 1.5, 2.0]
    best_score = -1
    best_clusters = None
    best_eps = None
    
    for eps in eps_values:
        clustering = DBSCAN(eps=eps, min_samples=3).fit(X_scaled)
        cluster_labels = clustering.labels_
        
        # Skip if too few clusters or too many noise points
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        if n_clusters < 2 or n_noise > len(cluster_labels) * 0.5:
            continue
            
        # Calculate silhouette score
        if n_clusters > 1:
            score = silhouette_score(X_scaled, cluster_labels)
            print(f"  eps={eps}: {n_clusters} clusters, {n_noise} noise points, silhouette={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_clusters = cluster_labels
                best_eps = eps
    
    return best_clusters, best_eps, best_score

def create_interactive_visualization(tsne_results, data):
    """Create interactive Plotly visualization."""
    print("📊 Creating interactive visualization...")
    
    # Use the best t-SNE result (typically perp_30)
    coords_key = 'perp_30' if 'perp_30' in tsne_results else list(tsne_results.keys())[0]
    coords = tsne_results[coords_key]
    
    # Create interactive scatter plot
    fig = px.scatter(
        x=coords[:, 0], 
        y=coords[:, 1],
        color=data['labels'],
        hover_data={
            'Crow Name': data['crow_names'],
            'Video': [os.path.basename(p) for p in data['video_paths']],
            'Frame': data['frame_numbers'],
            'Confidence': data['confidences']
        },
        title=f"Interactive t-SNE Visualization ({len(set(data['labels']))} Crows)",
        labels={'x': 't-SNE Dimension 1', 'y': 't-SNE Dimension 2', 'color': 'Crow ID'}
    )
    
    fig.update_layout(
        width=1000, 
        height=800,
        showlegend=len(set(data['labels'])) <= 20  # Hide legend if too many crows
    )
    
    # Save interactive plot
    interactive_path = os.path.join(OUTPUT_DIR, "interactive_tsne.html")
    fig.write_html(interactive_path)
    print(f"💻 Interactive plot saved: {interactive_path}")
    
    return fig

def identify_quality_issues(data, X_scaled):
    """Identify potential data quality issues."""
    print("🔍 Identifying potential quality issues...")
    
    issues = {
        'outliers': [],
        'duplicate_embeddings': [],
        'low_confidence': [],
        'single_video_crows': []
    }
    
    # 1. Outlier detection
    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(n_neighbors=min(20, len(X_scaled)//2))
    outlier_flags = lof.fit_predict(X_scaled)
    outlier_indices = np.where(outlier_flags == -1)[0]
    
    for idx in outlier_indices:
        issues['outliers'].append({
            'index': int(idx),
            'crow_id': data['labels'][idx],
            'crow_name': data['crow_names'][idx],
            'video': os.path.basename(data['video_paths'][idx]),
            'frame': data['frame_numbers'][idx],
            'confidence': data['confidences'][idx]
        })
    
    # 2. Nearly duplicate embeddings
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(X_scaled)
    for i in range(len(X_scaled)):
        for j in range(i+1, len(X_scaled)):
            if similarity_matrix[i, j] > 0.99:  # Very similar embeddings
                issues['duplicate_embeddings'].append({
                    'indices': [int(i), int(j)],
                    'similarity': float(similarity_matrix[i, j]),
                    'crow_ids': [data['labels'][i], data['labels'][j]]
                })
    
    # 3. Low confidence detections
    low_conf_threshold = np.percentile(data['confidences'], 10)  # Bottom 10%
    for idx, conf in enumerate(data['confidences']):
        if conf < low_conf_threshold:
            issues['low_confidence'].append({
                'index': int(idx),
                'crow_id': data['labels'][idx],
                'confidence': float(conf)
            })
    
    # 4. Crows from single video (potential overfitting)
    crow_video_counts = defaultdict(set)
    for idx, (crow_id, video) in enumerate(zip(data['labels'], data['video_paths'])):
        crow_video_counts[crow_id].add(video)
    
    for crow_id, videos in crow_video_counts.items():
        if len(videos) == 1:
            issues['single_video_crows'].append({
                'crow_id': crow_id,
                'video': list(videos)[0],
                'sample_count': data['labels'].count(crow_id)
            })
    
    return issues

def generate_report(data, tsne_results, clusters, quality_issues):
    """Generate comprehensive analysis report."""
    print("📋 Generating analysis report...")
    
    report = {
        'summary': {
            'total_embeddings': len(data['embeddings']),
            'unique_crows': len(set(data['labels'])),
            'unique_videos': len(set(data['video_paths'])),
            'avg_confidence': float(np.mean(data['confidences'])),
            'embedding_dimension': data['embeddings'].shape[1]
        },
        'crow_distribution': dict(Counter(data['labels'])),
        'tsne_results': {k: v.tolist() for k, v in tsne_results.items()},
        'quality_issues': quality_issues,
        'recommendations': []
    }
    
    # Generate recommendations
    if len(quality_issues['outliers']) > len(data['embeddings']) * 0.05:
        report['recommendations'].append("⚠️ High number of outliers detected - review detection quality")
    
    if len(quality_issues['single_video_crows']) > 0:
        report['recommendations'].append("⚠️ Some crows only appear in single videos - may indicate overfitting")
    
    if len(quality_issues['duplicate_embeddings']) > 0:
        report['recommendations'].append("⚠️ Near-duplicate embeddings found - check for data leakage")
    
    crow_counts = list(Counter(data['labels']).values())
    if max(crow_counts) / min(crow_counts) > 10:
        report['recommendations'].append("⚠️ Highly imbalanced dataset - consider data augmentation")
    
    if not report['recommendations']:
        report['recommendations'].append("✅ No major quality issues detected")
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, "analysis_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Analysis report saved: {report_path}")
    return report

def main():
    print("🎯 FACEBEAK EMBEDDING ANALYSIS")
    print("="*50)
    
    # Load data
    data = load_embeddings_with_metadata()
    if data is None:
        return
    
    # Run t-SNE analysis
    tsne_results, X_scaled = run_tsne_analysis(data)
    
    # Analyze clusters
    clusters, best_eps, silhouette = analyze_clusters(X_scaled, data['labels'])
    if clusters is not None:
        print(f"🎯 Best clustering: eps={best_eps}, silhouette score={silhouette:.3f}")
    
    # Create visualizations
    fig = create_interactive_visualization(tsne_results, data)
    
    # Traditional static plot
    coords = tsne_results[list(tsne_results.keys())[0]]
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=data['labels'], 
                   palette='tab20', legend=len(set(data['labels'])) <= 20)
    plt.title(f"t-SNE Visualization of Crow Embeddings ({len(set(data['labels']))} Crows)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tsne_scatter.png"), dpi=300)
    print("📊 Static plot saved: tsne_scatter.png")
    
    # Quality analysis
    quality_issues = identify_quality_issues(data, X_scaled)
    
    # Generate comprehensive report
    report = generate_report(data, tsne_results, clusters, quality_issues)
    
    print("\n" + "="*50)
    print("📊 ANALYSIS COMPLETE")
    print(f"✅ {report['summary']['total_embeddings']} embeddings analyzed")
    print(f"✅ {report['summary']['unique_crows']} unique crows")
    print(f"⚠️ {len(quality_issues['outliers'])} outliers detected")
    print(f"📁 Results saved to: {OUTPUT_DIR}/")
    print("="*50)

if __name__ == "__main__":
    main()
