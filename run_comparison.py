#!/usr/bin/env python3
"""
Comprehensive Algorithm Comparison
This script runs a single comparison and generates one comprehensive figure with 6 subplots
"""

import os
from algorithm_comparer import AlgorithmComparer


def run_single_comparison():
    """Run a single comprehensive algorithm comparison"""
    print("=== Running Comprehensive Algorithm Comparison ===")
    
    # Environment name
    env_name = "myoElbowPose1D6MRandom-v0"
    
    # Create algorithm comparer
    comparer = AlgorithmComparer(env_name)
    
    # Load or create policies for different algorithms
    algorithms = {
        "PPO": comparer.load_ppo_policy("models/ppo_model_100000_steps.zip_seed0.pt"),
        "Heuristic": comparer.create_simple_heuristic_policy(),
        "Zero": comparer.create_zero_policy(),
        "Random": comparer.create_random_policy()
    }
    
    print(f"Algorithms to compare: {', '.join(algorithms.keys())}")
    
    # Run algorithm comparison with multiple seeds
    results = comparer.compare_algorithms(
        algorithms=algorithms,
        seeds=5,  # Number of random seeds
        episodes_per_seed=10,  # Number of episodes per seed
        render=False
    )
    
    print("\n=== Comprehensive Algorithm Comparison Complete ===")
    print(f"Results saved to figures/comprehensive_multi_algorithm_comparison_{env_name}.png")
    
    # Close the comparer
    comparer.close()


def main():
    """Main function that runs the single comparison"""
    print("=== Running Single Algorithm Comparison ===")
    
    # Run single comprehensive comparison
    run_single_comparison()
    
    print("\n=== Comparison Complete ===")


if __name__ == "__main__":
    main()
