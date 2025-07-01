"""
Test ModelInterface with different batch size configurations
"""

import logging
import numpy as np
from pathlib import Path
import shutil
import time

# Import ModelInterface
from ModelInterface import ModelInterface

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("="*80)
print("Testing ModelInterface with different batch configurations")
print("="*80)

# Test configurations
configs = [
    {"max_batch": 32, "opt_batch": 16, "test_batch": 16},
    {"max_batch": 64, "opt_batch": 32, "test_batch": 32},
    {"max_batch": 128, "opt_batch": 64, "test_batch": 64},
    {"max_batch": 162, "opt_batch": 64, "test_batch": 162},  # FoundationPose default
]

# Clear cache first
cache_dir = Path("models/trt_cache")
if cache_dir.exists():
    print(f"\nClearing cache at {cache_dir}")
    shutil.rmtree(cache_dir)

for i, config in enumerate(configs):
    print(f"\n{'='*60}")
    print(f"Test {i+1}/{len(configs)}: max_batch={config['max_batch']}, "
          f"opt_batch={config['opt_batch']}, test_batch={config['test_batch']}")
    print("="*60)

    # Create unique cache directory for this config
    cache_suffix = f"_max{config['max_batch']}_opt{config['opt_batch']}"

    try:
        # Initialize interface
        interface = ModelInterface(
            max_batch_size=config['max_batch'],
            opt_batch_size=config['opt_batch'],
            use_fp16=True
        )

        # Modify cache paths to be unique
        orig_build = interface._build_or_load_engine
        def custom_build(onnx_path, engine_path, model_name):
            # Add suffix to engine path
            engine_path = engine_path.parent / f"{engine_path.stem}{cache_suffix}{engine_path.suffix}"
            return orig_build(onnx_path, engine_path, model_name)
        interface._build_or_load_engine = custom_build

        print("Loading models...")
        interface.load_models()

        # Create test data
        test_batch = config['test_batch']
        dummy_rgbddd = np.random.randn(test_batch, 160, 160, 6).astype(np.float32)
        dummy_rgbddd[:, :, :, :3] = np.clip(dummy_rgbddd[:, :, :, :3], 0, 1)
        dummy_poses = np.tile(np.eye(4), (test_batch, 1, 1)).astype(np.float32)

        # Test refinement
        print(f"Testing refinement with batch size {test_batch}...")
        t0 = time.time()
        refined = interface.refine_poses_batch(dummy_poses, dummy_rgbddd, dummy_rgbddd)
        t_refine = time.time() - t0
        print(f"✓ Refinement successful in {t_refine*1000:.1f} ms")

        # Test scoring
        print(f"Testing scoring with batch size {test_batch}...")
        t0 = time.time()
        scores = interface.score_poses_batch(dummy_rgbddd, dummy_rgbddd)
        t_score = time.time() - t0
        print(f"✓ Scoring successful in {t_score*1000:.1f} ms")
        print(f"  Scores shape: {scores.shape}, range: [{scores.min():.3f}, {scores.max():.3f}]")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("Batch configuration tests complete")
print("="*80)

# Cleanup
print("\nCleaning up test cache files...")
if cache_dir.exists():
    for f in cache_dir.glob("*"):
        if any(suffix in f.name for suffix in ["_max32_", "_max64_", "_max128_", "_max162_"]):
            f.unlink()
            print(f"  Removed {f.name}")
