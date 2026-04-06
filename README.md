# H-SACT: Heterogeneous Structure-Aware Link Prediction on Continuous-Time Heterogeneous Dynamic Graphs

This repository provides the official implementation of the **H-SACT** model for Temporal Graph Representation Learning and Link Property Prediction. 

A key feature of this framework is its custom **C++ Parallel Sampler** (`sampler_core`), which leverages OpenMP and `pybind11` to significantly accelerate the graph sampling process, enabling efficient training and inference on large-scale temporal graphs.

## Repository Structure

* `src/` : Contains the core source code, including the H-SACT model architectures, utility functions, and the main training pipeline (`main.py`).
    * `src/utils/sampler_core.cpp`: The underlying C++ parallel sampler used for fast dynamic graph operations.
* `scripts/` : Shell scripts for automating tasks and running batch experiments.
* `tgb/` : Modules for Temporal Graph Benchmark (TGB) data processing and evaluation.

## Environment Setup

To run the H-SACT framework, you need a Python environment (>= 3.9) and a C++ compiler supporting OpenMP (e.g., GCC).

### 1. Create Environment
It is recommended to use Conda to manage your dependencies:
```bash
conda create -n hsact_env python=3.9
conda activate hsact_env
```

### 2. Install Dependencies
Ensure you have installed PyTorch and PyTorch Geometric (PyG) compatible with your CUDA version. Then, install `pybind11` which is strictly required to compile the C++ sampling extension:
```
pip install pybind11>=2.5
```

### 3. Compile and Install the Project
You must compile the `sampler_core` module before running the experiments. Run the following command in the root directory:
```bash
pip install -e .
```

> **Note**: If the standard installation fails to build the C++ extension due to environment isolation, please try the following alternative:
> ```bash
> pip install --no-build-isolation -e .
> ```

## Running Examples

The main entry point for training and testing the H-SACT model is `src/main.py`. You can run the code using the provided shell scripts or directly via the command line.

### Method 1: Using Shell Scripts
We provide batch execution scripts in the `scripts/` directory to facilitate running multiple datasets sequentially or in parallel. 

```bash
# Run the Binary Cross Entropy (BCE) experiment script
bash scripts/bce.sh
```
*Note: You may need to edit `scripts/bce.sh` to update the `run_python` variable to match your local python path.*

### Method 2: Manual Command Line Execution
For finer control over the hyperparameters, you can execute the main script directly. Here is an example command to run H-SACT on the `thgl-forum-subset` dataset using GPU 0:

```bash
python src/main.py \
    --exper_name hsact_default_run \
    --dataset thgl-forum-subset \
    --use_onehot_node_feats \
    --use_graph_structure \
    --use_gpu 1 \
    --device 0
```

### Key Arguments Explained
* `--exper_name`: Defines the name of the current experiment. Logs, model checkpoints, and result files will be organized under `exper_base_dir/<exper_name>/<dataset>/`.
* `--dataset`: Specifies the temporal graph dataset to use (e.g., `thgl-forum-subset`, `thgl-github-subset`, `thgl-myket-subset`, `thgl-software-subset`).
* `--use_onehot_node_feats`: Enables the use of one-hot node features during training.
* `--use_graph_structure`: Instructs the model to incorporate structural graph information.
* `--use_gpu`: Set to `1` to train on GPU, or `0` to fall back to CPU.
* `--device`: Determines the specific GPU ID to use. Setting this to `-1` will automatically utilize all available GPUs via `DataParallel`.

## Results and Evaluation
After executing an experiment (e.g., `<exper_name>`), all related files are systematically organized under the specific dataset directory (`exper/<exper_name>/<dataset>/`):

1. **Execution Logs:** Detailed training logs and configurations are saved directly in the dataset folder as `<exper_name>.log`.
2. **Model Checkpoints:** The trained model weights for each run are saved as `.pt` files in the `checkpoint/` directory (formatted as `<dataset>_<seed>_<run_idx>.pt`).
3. **Raw Predictions:** The prediction scores for validation and test sets across different runs are exported as `.npz` files within the `output/run_<idx>/` subdirectories.
4. **Performance Metrics:** The final aggregated evaluation metrics (such as mean and standard deviation across all runs) are stored in JSON format at `result/<dataset>_results.json`.
