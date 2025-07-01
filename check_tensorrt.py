"""
Check TensorRT installation and GPU compatibility
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

print("="*60)
print("TensorRT Diagnostic Information")
print("="*60)

# TensorRT version
print(f"\nTensorRT version: {trt.__version__}")

# CUDA info
cuda.init()
device = cuda.Device(0)
print(f"\nGPU: {device.name()}")
print(f"Compute Capability: {device.compute_capability()}")
print(f"Total Memory: {device.total_memory() / (1024**3):.1f} GB")

# Get free memory
free, total = cuda.mem_get_info()
print(f"Free Memory: {free / (1024**3):.1f} GB")
print(f"Used Memory: {(total - free) / (1024**3):.1f} GB")

# TensorRT logger levels
print(f"\nTensorRT Logger Levels:")
print(f"  VERBOSE: {trt.Logger.VERBOSE}")
print(f"  INFO: {trt.Logger.INFO}")
print(f"  WARNING: {trt.Logger.WARNING}")
print(f"  ERROR: {trt.Logger.ERROR}")
print(f"  INTERNAL_ERROR: {trt.Logger.INTERNAL_ERROR}")

# Check if we can create a simple engine
print("\nTesting simple TensorRT engine creation...")
try:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # Create a simple input
    input_tensor = network.add_input("input", trt.float32, (1, 3, 32, 32))

    # Add a simple convolution
    conv_weights = np.random.randn(8, 3, 3, 3).astype(np.float32)
    conv_layer = network.add_convolution_nd(input_tensor, 8, (3, 3), conv_weights)
    conv_layer.stride_nd = (1, 1)
    conv_layer.padding_nd = (1, 1)

    # Mark output
    network.mark_output(conv_layer.get_output(0))

    # Build engine
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    print("Building test engine...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine:
        print("✓ Test engine built successfully")
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        print(f"✓ Engine deserialized successfully")
        print(f"  Num bindings: {engine.num_bindings}")
        print(f"  Max batch size: {engine.max_batch_size}")
    else:
        print("✗ Failed to build test engine")

except Exception as e:
    print(f"✗ Error during engine test: {e}")

print("\n" + "="*60)
