#!/usr/bin/env python3
"""
Workflow Checker for Unsupervised Learning
Verifies that necessary data is available before running unsupervised techniques.
"""

import sys
from pathlib import Path
import logging
from db import get_all_crows, get_crow_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_data_readiness() -> bool:
    """
    Check if the necessary data is available for unsupervised learning.
    
    Returns:
        True if ready, False otherwise
    """
    logger.info("🔍 Checking data readiness for unsupervised learning...")
    
    issues = []
    warnings = []
    
    # 1. Check if we have any crows in the database
    crows = get_all_crows()
    if not crows:
        issues.append("❌ No crows found in database. Run video processing first!")
        return False
    
    logger.info(f"✅ Found {len(crows)} crows in database")
    
    # 2. Check if crows have embeddings
    crows_with_embeddings = 0
    total_embeddings = 0
    crows_with_temporal = 0
    
    for crow in crows:
        embeddings = get_crow_embeddings(crow['id'])
        if embeddings:
            crows_with_embeddings += 1
            total_embeddings += len(embeddings)
            
            # Check if embeddings have temporal metadata
            temporal_count = sum(1 for emb in embeddings 
                               if emb.get('frame_number') is not None 
                               and emb.get('video_path'))
            if temporal_count > 0:
                crows_with_temporal += 1
    
    if crows_with_embeddings == 0:
        issues.append("❌ No embeddings found. Run detection and embedding extraction first!")
        return False
    
    logger.info(f"✅ {crows_with_embeddings}/{len(crows)} crows have embeddings")
    logger.info(f"✅ Total embeddings: {total_embeddings}")
    
    if crows_with_temporal < crows_with_embeddings:
        warnings.append(
            f"⚠️ Only {crows_with_temporal}/{crows_with_embeddings} crows have temporal metadata. "
            "Temporal consistency loss may not work optimally."
        )
    
    # 3. Check minimum data requirements
    if total_embeddings < 50:
        warnings.append(
            f"⚠️ Only {total_embeddings} embeddings available. "
            "Consider processing more videos for better unsupervised learning results."
        )
    
    if len(crows) < 3:
        warnings.append(
            f"⚠️ Only {len(crows)} unique crows. "
            "Merge suggestions and clustering will be limited."
        )
    
    # 4. Check for crop image directories
    crop_dirs = [
        Path("crow_crops"),
        Path("videos"),
        Path("extracted_crops")
    ]
    
    existing_crop_dirs = [d for d in crop_dirs if d.exists() and any(d.glob("*.jpg"))]
    if existing_crop_dirs:
        total_crops = sum(len(list(d.glob("**/*.jpg"))) for d in existing_crop_dirs)
        logger.info(f"✅ Found {total_crops} crop images in {len(existing_crop_dirs)} directories")
    else:
        warnings.append("⚠️ No crop image directories found. SimCLR pretraining may not be available.")
    
    # Print warnings
    for warning in warnings:
        logger.warning(warning)
    
    if issues:
        logger.error("❌ Cannot proceed with unsupervised learning:")
        for issue in issues:
            logger.error(f"  {issue}")
        return False
    
    logger.info("🎉 Data is ready for unsupervised learning!")
    return True


def suggest_next_steps():
    """Suggest next steps based on current data state."""
    
    if not check_data_readiness():
        logger.info("\n📋 Suggested next steps for crop extraction:")
        logger.info("1. Use the extraction GUI (recommended):")
        logger.info("   python extract_training_gui.py")
        logger.info("2. Or extract via command line:")
        logger.info("   python main.py --mode process_video --video_path your_video.mp4")
        logger.info("3. Or use the dedicated extraction script:")
        logger.info("   python extract_training_data.py --video your_video.mp4")
        logger.info("4. Then return to run unsupervised learning")
        return
    
    logger.info("\n🚀 You're ready! Suggested workflow:")
    logger.info("1. Try the GUI tools first:")
    logger.info("   python unsupervised_gui_tools.py")
    logger.info("2. Or run enhanced training:")
    logger.info("   python train_with_unsupervised.py --epochs 50")
    logger.info("3. Check the guide for detailed instructions:")
    logger.info("   UNSUPERVISED_LEARNING_GUIDE.md")


if __name__ == "__main__":
    suggest_next_steps() 