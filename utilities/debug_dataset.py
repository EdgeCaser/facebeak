#!/usr/bin/env python3
"""
Debug script to test the CrowTripletDataset
"""

import sys
import torch
from old_scripts.train_triplet_resnet import CrowTripletDataset, custom_triplet_collate
from torch.utils.data import DataLoader

def test_dataset():
    print("Testing CrowTripletDataset...")
    
    try:
        # Create dataset
        dataset = CrowTripletDataset('crow_crops', audio_dir=None, split='train')
        print(f"Dataset created successfully with {len(dataset)} samples")
        print(f"Number of crows: {len(dataset.crow_to_imgs)}")
        print(f"Crow IDs: {list(dataset.crow_to_imgs.keys())[:5]}...")  # Show first 5
        
        # Test a single sample
        print("\nTesting single sample...")
        sample = dataset[0]
        print(f"Sample type: {type(sample)}")
        print(f"Sample length: {len(sample)}")
        
        if len(sample) == 3:
            imgs, audio, crow_id = sample
            print(f"Images type: {type(imgs)}")
            print(f"Audio type: {type(audio)}")
            print(f"Crow ID: {crow_id}")
            
            if isinstance(imgs, tuple) and len(imgs) == 3:
                print(f"Image shapes: {[img.shape if img is not None else None for img in imgs]}")
            else:
                print(f"Images not a tuple of 3: {imgs}")
                
            if isinstance(audio, tuple) and len(audio) == 3:
                print(f"Audio: {[a is not None for a in audio]}")
            else:
                print(f"Audio not a tuple of 3: {audio}")
        else:
            print(f"Sample doesn't have 3 elements: {sample}")
            
        # Test DataLoader with custom collate function
        print("\nTesting DataLoader with custom collate...")
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=custom_triplet_collate)
        
        print("Getting first batch...")
        batch = next(iter(loader))
        print(f"Batch type: {type(batch)}")
        print(f"Batch length: {len(batch)}")
        
        if len(batch) == 3:
            batch_imgs, batch_audio, batch_ids = batch
            print(f"Batch images type: {type(batch_imgs)}")
            print(f"Batch audio type: {type(batch_audio)}")
            print(f"Batch IDs: {batch_ids}")
            
            if isinstance(batch_imgs, tuple) and len(batch_imgs) == 3:
                anchor, pos, neg = batch_imgs
                print(f"Batch image shapes: anchor={anchor.shape}, pos={pos.shape}, neg={neg.shape}")
        
        print("SUCCESS: DataLoader with custom collate works!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset() 