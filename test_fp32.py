"""
Test ModelInterface with FP32 precision to diagnose Cask errors
"""

import logging
import numpy as np
from pathlib import Path
import shutil

# Import ModelInterface
from ModelInterface import ModelInterface

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("="*80)
print("Testing ModelInterface with FP32 precision")
print("="*80)

# First, clear any existing FP16 cache
cache_dir = Path("models/trt_cache")
if cache_dir.exists():
    print(f"\nClearing existing cache at {cache_dir}")
    shutil.rmtree(cache_dir)
    print("✓ Cache cleared")

# Test with FP32
print("\n" + "-"*60)
print("Testing with FP32 precision")
print("-"*60)

try:
    interface_fp32 = ModelInterface(
        max_batch_size=64,  # Smaller batch for testing
        opt_batch_size=32,
        use_fp16=False  # Disable FP16
    )

    print("Loading models...")
    interface_fp32.load_models()

    print("Running warmup...")
    interface_fp32.warmup()

    print("\nTesting inference...")
    # Create test data
    dummy_rgbddd = np.random.randn(32, 160, 160, 6).astype(np.float32)
    dummy_rgbddd[:, :, :, :3] = np.clip(dummy_rgbddd[:, :, :, :3], 0, 1)
    dummy_poses = np.tile(np.eye(4), (32, 1, 1)).astype(np.float32)

    # Test refinement
    print("Testing refinement...")
    refined = interface_fp32.refine_poses_batch(dummy_poses, dummy_rgbddd, dummy_rgbddd)
    print(f"✓ Refinement successful, output shape: {refined.shape}")

    # Test scoring
    print("Testing scoring...")
    scores = interface_fp32.score_poses_batch(dummy_rgbddd, dummy_rgbddd)
    print(f"✓ Scoring successful, output shape: {scores.shape}")
    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    print("\n✓ FP32 test completed successfully!")

except Exception as e:
    print(f"\n✗ FP32 test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
