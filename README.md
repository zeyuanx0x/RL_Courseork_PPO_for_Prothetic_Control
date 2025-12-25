# MyoSuite Reinforcement Learning Framework

A comprehensive reinforcement learning framework for musculoskeletal control using MyoSuite, focusing on Proximal Policy Optimization (PPO) and comparative analysis with other algorithms.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Running Experiments](#running-experiments)
- [Results Analysis](#results-analysis)
- [Visualization](#visualization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This framework provides a complete pipeline for:

- **Training RL algorithms** on MyoSuite musculoskeletal environments
- **Comparing algorithms** across multiple performance metrics
- **Evaluating robustness** under various perturbations
- **Analyzing physiology** and metabolic costs
- **Optimizing exoskeleton assistance** levels

### Key Features

- ✅ Support for multiple algorithms (PPO, SAC, Traditional Control, Heuristic)
- ✅ Multi-seed training and evaluation
- ✅ Domain randomization and perturbation testing
- ✅ Metabolic and muscle activation analysis
- ✅ Exoskeleton assistance optimization
- ✅ Publication-quality visualization
- ✅ Comprehensive statistical analysis
- ✅ Easy-to-use command-line interface
- ✅ Modular, extensible design

### Supported Environments

The framework is primarily tested on:
- `myoElbowPose1D6MRandom-v0` - 1D elbow pose control with 6 muscles

But can be extended to other MyoSuite environments.

## Installation

### Prerequisites

- Python 3.8+ with pip
- PyTorch 1.8+ (with CUDA support recommended)
- MyoSuite and its dependencies

## Quick Start

### Run the Complete Pipeline

#### 1. Install Dependencies

First, install all required dependencies:

```bash
pip install -r requirements.txt
```

#### 2. Train PPO Agent

##### Basic Training
Train a PPO agent with default parameters:

```bash
python main.py --experiment-type ppo-train
```

##### Customized Training
Train with custom parameters and multiple seeds for more robust results:

```bash
# Train with specific hyperparameters
python main.py --experiment-type ppo-train \
    --env-name myoElbowPose1D6MRandom-v0 \
    --seed 42 \
    --total-timesteps 1000000 \
    --lr 3e-4 \
    --clip-range 0.2

# Train with multiple seeds (recommended for robust results)
python main.py --experiment-type ppo-train --num-seeds 5
```

#### 3. Evaluate Trained Policies

Evaluate the trained PPO policies on the default environment:

```bash
# Evaluate a single trained model
python main.py --experiment-type ppo-eval \
    --load-model checkpoints/ppo_myoElbowPose1D6MRandom-v0_seed42.pt

# Evaluate all trained models from multi-seed training
python main.py --experiment-type ppo-eval --evaluate-all
```

#### 4. Run Robustness Testing

Test the trained policies under various perturbations:

```bash
python robustness_tester.py --load-models checkpoints/ --env-name myoElbowPose1D6MRandom-v0
```

#### 5. Generate Results

##### Generate All Results

Run the complete results generation pipeline:

```bash
python run_results.py
```

##### Generate Specific Results

Generate only the results you need:

```bash
# Generate all results (excluding figures)
python generate_results.py --all --no-figures

# Generate hyperparameter tuning results
python generate_results.py --hyperparam --no-figures

# Generate robustness results
python generate_results.py --robustness --no-figures

# Generate physiology results
python generate_results.py --physiology --no-figures

# Generate algorithm comparison results
python generate_results.py --algorithm --no-figures

# Generate exoskeleton results
python generate_results.py --exoskeleton --no-figures
```

#### 6. Run Statistical Analysis

Perform statistical analysis on the generated results:

```bash
python statistical_report.py --input-dir results/ --output-dir results/
```

#### 7. Complete Pipeline with One Command

Alternatively, execute the full experiment pipeline with a single command:

```bash
python all_results.py --no-figures
```

This will:
1. Train PPO agent on the default environment with multiple seeds
2. Evaluate the trained policies across all seeds
3. Perform robustness testing under various perturbations
4. Generate all results (excluding figures)
5. Perform statistical analysis
6. Create comprehensive reports

#### 8. Verify Generated Results

Check the generated results in the output directories:

```bash
# List all generated result files
ls -la results/

# Check the combined results CSV
cat results/all_results.csv

# View the statistical report
cat results/statistical_report.md
```

### Pipeline Output Files

All generated results will be stored in the following locations:

- **`checkpoints/`**: Trained model weights (e.g., `ppo_myoElbowPose1D6MRandom-v0_seed42.pt`)
- **`results/all_results.csv`**: Combined results from all experiments
- **`results/statistical_report.md`**: Statistical analysis summary
- **`results/academic_report.md`**: Comprehensive academic report
- **`reports/RESULTS.md`**: High-level results summary
- **`results/`**: Raw data files for each experiment type

### Generate Results Only

If you already have trained models, you can generate results and figures without retraining:

```bash
python run_results.py --only-report
```

### Generate Specific Results

Generate only the results you need:

```bash
# Generate all results
python generate_results.py --all

# Generate hyperparameter tuning results
python generate_results.py --hyperparam

# Generate robustness results
python generate_results.py --robustness

# Generate physiology results
python generate_results.py --physiology

# Generate algorithm comparison results
python generate_results.py --algorithm

# Generate exoskeleton results
python generate_results.py --exoskeleton

# Generate rendering results
python generate_results.py --rendering
```

## Usage Guide

### Training a PPO Agent

Train a PPO agent with default parameters:

```bash
python main.py --experiment-type ppo-train
```

### Customize Training Parameters

```bash
python main.py --experiment-type ppo-train \
    --env-name myoElbowPose1D6MRandom-v0 \
    --seed 42 \
    --total-timesteps 500000 \
    --lr 3e-4 \
    --clip-range 0.2 \
    --gamma 0.99 \
    --lambda-gae 0.95
```

### Multi-Seed Training

Train with multiple random seeds for more robust results:

```bash
python main.py --experiment-type ppo-train --num-seeds 5
```

### Evaluate a Trained Model

```bash
python main.py --experiment-type ppo-eval \
    --load-model checkpoints/ppo_myoElbowPose1D6MRandom-v0_seed42.pt
```

### Environment Inspection

Visualize and inspect the environment:

```bash
python main.py --experiment-type env-inspect --render
```

## Project Structure

```
RL_PPO_Coursework/
├── checkpoints/          # Trained model weights
├── configs/              # Experiment configurations
├── figures/              # All generated figures
├── reports/              # Result reports
├── results/              # Raw results data and academic report
├── utils/                # Utility functions
│   ├── env_utils.py      # Environment compatibility utilities
│   ├── eval_utils.py     # Policy evaluation utilities
│   ├── logging_utils.py  # Logging and directory setup
│   ├── plotting_utils.py # Plotting utilities
│   └── render_utils.py   # Rendering utilities
├── experiment_manager.py # Experiment orchestration
├── ppo_agent.py          # PPO implementation
├── generate_results.py   # Results generation
├── robustness_tester.py  # Robustness testing
├── statistical_report.py # Statistical analysis
├── run_results.py        # Main pipeline runner
├── main.py               # Interactive training interface
├── README.md             # This file
```

## Configuration

### Main Configuration File

Configuration parameters are managed through command-line arguments. Key parameters include:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--env-name` | MyoSuite environment name | `myoElbowPose1D6MRandom-v0` |
| `--seed` | Random seed | `42` |
| `--total-timesteps` | Training timesteps | `1000000` |
| `--lr` | Learning rate | `3e-4` |
| `--clip-range` | PPO clip range | `0.2` |
| `--gamma` | Discount factor | `0.99` |
| `--lambda-gae` | GAE lambda parameter | `0.95` |

### Output Directories

| Directory | Purpose |
|-----------|---------|
| `checkpoints/` | Trained model weights |
| `figures/` | All generated figures |
| `reports/` | Result summaries |
| `results/` | Raw data, CSV files, and academic report |



## Statistical Analysis

The framework performs comprehensive statistical analysis including:

- Mann-Whitney U test for distribution comparison
- Cohen's d for effect size calculation
- Cliff's delta for magnitude of difference
- Confidence interval estimation


## Visualization

### Figures

All figures are saved to the `figures/` directory with descriptive filenames. Key figures include:

1. **Training Curves**: Learning progress over time
2. **Algorithm Comparison**: Performance across algorithms
3. **Robustness Analysis**: Performance under perturbations
4. **Physiology Analysis**: Muscle activation and energy usage
5. **Exoskeleton Assistance**: Optimal assistance level identification
6. **Value Heatmaps**: Learned value function visualization
7. **Rendering Results**: Rendering quality comparison


## Troubleshooting

### Common Issues

1. **MyoSuite installation errors**:
   - Ensure you have installed all MyoSuite dependencies
   - Verify that MuJoCo is properly set up
   - Check compatibility with your Python version

2. **CUDA errors during training**:
   - Ensure PyTorch with CUDA support is installed
   - Check if your GPU is compatible
   - Try running with CPU only: `python main.py --experiment-type ppo-train --device cpu`

3. **Memory issues**:
   - Reduce batch size or rollout length
   - Limit the number of seeds for multi-seed training
   - Close other GPU-intensive applications

4. **Rendering issues**:
   - Ensure you have the required display drivers
   - Try running without rendering: `python main.py --experiment-type ppo-train --no-render`

### Logs and Debugging

- Training logs are printed to the console
- Error messages include detailed stack traces
- Debug information can be enabled by adding `--verbose` flag

## Contributing

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Run tests** to ensure nothing is broken
5. **Submit a pull request**

## Acknowledgments

- MyoSuite development team for the musculoskeletal simulation framework
- Open-source RL community for algorithm implementations

## Contact

For questions, issues, or collaboration opportunities, please contact:

- Project maintainer: [Zeyuan Xin]
- Email: [izer0x@outlook.com]

---