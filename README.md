# Run this Repo

First you will need to get access to the data and onnx models they can be found in onedrive at:

- [Data](https://sarcos-my.sharepoint.us/:u:/g/personal/joshua_powers_palladyneai_com/EexmUSPFbthCnvXQu8hdRG8BAvuva_5_1vNGZSZLct16ig?e=3jXsV7)
- [ONNX Models](https://sarcos-my.sharepoint.us/:f:/g/personal/joshua_powers_palladyneai_com/EttXL5SHaxRLpcJEjvnn6bYBHBj6xERsWwnavIYuQUtehg?e=CA3K9o)

Put both of these folder in the root of the repo and uncompress the data folder as data.

#### Sync uv with pyproject.toml to install dependencies

```bash
uv sync
```

#### Run Demo

```bash
uv run run_demo.py
```

Note: the first time you run this it will take a very long time as the onnx models are being compiled into optimized TensorRT model for your specific system setup, after the first run the TensorRT engine files will be caches in models/trt_cache, if you want to rebuild the enginef iles delete that folder. Engine files are device specific so they must be built on the device you intend to run the deom on.
