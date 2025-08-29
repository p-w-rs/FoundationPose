# utils/model.py
import tensorrt as trt
import cupy as cp
import os

class ModelTRT:
    def __init__(self, model_name, output_shapes, onnx_path=None, opt_level=5, min_batch=1, opt_batch=252, max_batch=252, fp16=True):
        """
        Generic TensorRT model wrapper

        Args:
            model_name: Name of the model (e.g., "refine_model", "score_model")
            output_shapes: List of functions that return output shape given batch_size
                          e.g., [lambda b: (b, 3), lambda b: (b, 3)] for refine
                          or   [lambda b: (1, b)] for score
            onnx_path: Path to ONNX model (defaults to models/{model_name}.onnx)
        """
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.model_name = model_name
        self.output_shapes = output_shapes
        self.num_outputs = len(output_shapes)

        if onnx_path is None:
            onnx_path = f"models/{model_name}.onnx"

        base_dir = os.path.dirname(onnx_path)
        engine_dir = os.path.join(base_dir, "trt_cache")
        engine_path = os.path.join(engine_dir, f"{model_name}_{min_batch}_{opt_batch}_{max_batch}_{fp16}.engine")
        os.makedirs(engine_dir, exist_ok=True)

        # Load or build engine
        if os.path.exists(engine_path):
            with open(engine_path, "rb") as f:
                self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        else:
            self.engine = self._build_engine(onnx_path, engine_path, opt_level, min_batch, opt_batch, max_batch, fp16)

        self.context = self.engine.create_execution_context()
        self.stream = cp.cuda.Stream()

    def _build_engine(self, onnx_path, engine_path, opt_level, min_batch, opt_batch, max_batch, fp16):
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)

        with open(onnx_path, "rb") as f:
            parser.parse(f.read())

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        config.builder_optimization_level = opt_level

        profile = builder.create_optimization_profile()
        for name in ["input1", "input2"]:
            profile.set_shape(name, (min_batch, 160, 160, 6), (opt_batch, 160, 160, 6), (max_batch, 160, 160, 6))
        config.add_optimization_profile(profile)

        engine_bytes = builder.build_serialized_network(network, config)
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        return trt.Runtime(self.logger).deserialize_cuda_engine(engine_bytes)

    def exec(self, input1: cp.ndarray, input2: cp.ndarray, outputs: list):
        """
        Execute the model with CuPy arrays

        Args:
            input1, input2: CuPy arrays of shape (batch, 160, 160, 6)
            outputs: Optional list of pre-allocated output arrays

        Returns:
            List of output arrays (or single array if only one output)
        """
        # Set input shapes
        self.context.set_input_shape("input1", tuple(input1.shape))
        self.context.set_input_shape("input2", tuple(input2.shape))

        # Set tensor addresses using CuPy's data.ptr
        self.context.set_tensor_address("input1", input1.data.ptr)
        self.context.set_tensor_address("input2", input2.data.ptr)

        # Set output addresses
        for i, output in enumerate(outputs):
            self.context.set_tensor_address(f"output{i+1}", output.data.ptr)

        # Execute
        with self.stream:
            self.context.execute_async_v3(self.stream.ptr)
        self.stream.synchronize()

        # Return single output if only one, otherwise return list
        return outputs


class OptimizedModelTRT(ModelTRT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_buffers = {}

    def get_output_buffers(self, batch_size):
        """Get or create output buffers for given batch size"""
        if batch_size not in self.output_buffers:
            outputs = []
            for shape_fn in self.output_shapes:
                shape = shape_fn(batch_size)
                outputs.append(cp.empty(shape, dtype=cp.float32))
            self.output_buffers[batch_size] = outputs
        return self.output_buffers[batch_size]

    def exec_optimized(self, input1, input2):
        """Execute with automatic buffer management"""
        batch_size = input1.shape[0]
        outputs = self.get_output_buffers(batch_size)
        return self.exec(input1, input2, outputs)


def benchmark_model(model, model_name, batch_sizes=[64, 128, 252]):
    import time
    import numpy as np
    """Benchmark a model with various batch sizes"""
    print(f"\nBenchmarking {model_name}:")

    for batch_size in batch_sizes:
        # Create CuPy arrays on GPU
        input1 = cp.random.randn(batch_size, 160, 160, 6, dtype=cp.float32)
        input2 = cp.random.randn(batch_size, 160, 160, 6, dtype=cp.float32)

        # Pre-allocate outputs
        outputs = []
        for shape_fn in model.output_shapes:
            shape = shape_fn(batch_size)
            outputs.append(cp.empty(shape, dtype=cp.float32))

        # Warmup
        for _ in range(10):
            model.exec(input1, input2, outputs)

        # Benchmark
        cp.cuda.Device().synchronize()
        times = []
        for _ in range(100):
            start = time.perf_counter()
            result = model.exec(input1, input2, outputs)
            cp.cuda.Device().synchronize()
            times.append((time.perf_counter() - start) * 1000)

        print(f"  Batch {batch_size}: {np.mean(times):.2f} ± {np.std(times):.2f} ms")

        # Verify output shapes
        if isinstance(result, list):
            for i, out in enumerate(result):
                print(f"    Output {i+1} shape: {out.shape}")
        else:
            print(f"    Output shape: {result.shape}")


if __name__ == "__main__":
    # Initialize models
    refine_model = ModelTRT(
        model_name="refine_model",
        output_shapes=[
            lambda b: (b, 3),  # output1
            lambda b: (b, 3)   # output2
        ]
    )

    score_model = ModelTRT(
        model_name="score_model",
        output_shapes=[
            lambda b: (1, b)   # output1
        ]
    )

    # Benchmark both models
    benchmark_model(refine_model, "Refine Model")
    benchmark_model(score_model, "Score Model")
