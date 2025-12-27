# PPO for Myoelectric Elbow Control

A Proximal Policy Optimization (PPO) implementation for musculoskeletal control using MyoSuite, focusing on elbow joint control with myoelectric signals.

## Overview

This repository provides a high-quality implementation of PPO for training reinforcement learning agents on MyoSuite environments, specifically for myoelectric elbow control. The implementation includes utilities for training, evaluation, and analysis of musculoskeletal control policies.

## Prerequisites

- Python 3.8+
- PyTorch 1.8+ (CUDA support recommended)
- MyoSuite and its dependencies

## Installation

```bash
# Clone the repository
git clone https://github.com/zeyuanx0x/PPO-for-myoeletric-elbow-control_RL-Coursework.git
cd PPO-for-myoeletric-elbow-control_RL-Coursework

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training

Train a PPO agent on the default environment (`myoElbowPose1D6MRandom-v0`) with default parameters:

```bash
python main.py --experiment-type ppo-train
```

Train with custom hyperparameters:

```bash
python main.py --experiment-type ppo-train \
    --env-name myoElbowPose1D6MRandom-v0 \
    --seed 42 \
    --total-timesteps 1000000 \
    --lr 3e-4 \
    --clip-range 0.2
```

Train with multiple seeds for robust results:

```bash
python main.py --experiment-type ppo-train --num-seeds 5
```

### Evaluation

Evaluate a trained model:

```bash
python main.py --experiment-type ppo-eval \
    --load-model checkpoints/ppo_myoElbowPose1D6MRandom-v0_seed42.pt
```

Evaluate all trained models from multi-seed training:

```bash
python main.py --experiment-type ppo-eval --evaluate-all
```

### Environment Inspection

Visualize and inspect the environment:

```bash
python main.py --experiment-type env-inspect --render
```

## Key Features

- **PPO Implementation**: High-quality PPO algorithm implementation for musculoskeletal control
- **Multi-Seed Training**: Support for training with multiple random seeds for robust results
- **MyoSuite Integration**: Seamless integration with MyoSuite musculoskeletal environments
- **Evaluation Tools**: Comprehensive evaluation utilities for trained policies
- **Robustness Testing**: Tools for testing policy robustness under perturbations
- **Physiology Analysis**: Metabolic and muscle activation analysis
- **Exoskeleton Optimization**: Utilities for optimizing exoskeleton assistance levels
- **Publication-Quality Visualization**: Tools for generating high-quality figures

## Supported Environments

The implementation is primarily tested on:
- `myoElbowPose1D6MRandom-v0` - 1D elbow pose control with 6 muscles

It can be extended to other MyoSuite environments with minimal modifications.

## Advanced Usage

### Robustness Testing

Test trained policies under various perturbations:

```bash
python robustness_tester.py --load-models checkpoints/ --env-name myoElbowPose1D6MRandom-v0
```

### Results Generation

Generate results from experiments:

```bash
# Generate all results
python generate_results.py --all

# Generate specific result types
python generate_results.py --robustness
python generate_results.py --physiology
python generate_results.py --algorithm
```

### Statistical Analysis

Perform statistical analysis on results:

```bash
python statistical_report.py --input-dir results/ --output-dir results/
```

### Complete Pipeline

Run the complete experiment pipeline with one command:

```bash
python all_results.py --no-figures
```

## Troubleshooting

### Common Issues

1. **MyoSuite installation errors**:
   - Ensure all MyoSuite dependencies are installed
   - Verify MuJoCo is properly set up
   - Check Python version compatibility

2. **CUDA errors**:
   - Ensure PyTorch with CUDA support is installed
   - Try running on CPU: `python main.py --experiment-type ppo-train --device cpu`

3. **Memory issues**:
   - Reduce batch size or rollout length
   - Limit the number of seeds for multi-seed training

4. **Rendering issues**:
   - Ensure display drivers are up to date
   - Try running without rendering: `python main.py --experiment-type ppo-train --no-render`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- MyoSuite development team for the musculoskeletal simulation framework
- Open-source RL community for algorithm implementations

## Contact

For questions or issues, please contact:
- Project maintainer: Zeyuan Xin
- Email: izer0x@outlook.com

## License

MIT License

---