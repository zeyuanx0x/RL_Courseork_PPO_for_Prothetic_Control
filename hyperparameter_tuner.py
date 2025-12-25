import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Dict, List, Any
from myosuite.utils import gym
from ppo_agent import PPOAgent


class HyperparameterTuner:
    """Hyperparameter tuner for PPO algorithm"""
    
    def __init__(self, env_name: str):
        self.env_name = env_name
    
    def tune_hyperparameters(
        self,
        hyperparameter_grid: Dict[str, List],
        total_timesteps: int = 100000,
        rollout_length: int = 2048,
        episodes_per_tune: int = 5
    ) -> Dict[str, Any]:
        """Hyperparameter tuning using grid search
        
        Args:
            hyperparameter_grid: Hyperparameter grid to search, format {param_name: [values]}
            total_timesteps: Total timesteps per tuning run
            rollout_length: Rollout length per update
            episodes_per_tune: Number of episodes to evaluate per tuning run
            
        Returns:
            Tuning results with best hyperparameters and performance
        """
        print(f"=== Hyperparameter Tuning: {self.env_name} ===")
        print(f"Grid search with {self._calculate_grid_size(hyperparameter_grid)} combinations")
        
        # Generate all hyperparameter combinations
        combinations = self._generate_combinations(hyperparameter_grid)
        
        results = {
            "combinations": [],
            "returns": [],
            "episode_lengths": [],
            "best_params": None,
            "best_return": -float('inf')
        }
        
        # Evaluate each combination
        for i, params in enumerate(combinations):
            print(f"\nTesting combination {i+1}/{len(combinations)}: {params}")
            
            # Create environment and agent
            env = gym.make(self.env_name)
            agent = PPOAgent(
                env,
                gamma=params.get("gamma", 0.99),
                lambda_gae=params.get("lambda_gae", 0.95),
                clip_range=params.get("clip_range", 0.2),
                lr=params.get("lr", 3e-4),
                seed=42
            )
            
            # Run training
            stats = agent.run_training(
                total_timesteps=total_timesteps,
                rollout_length=rollout_length,
                render=False
            )
            
            # Evaluate performance
            avg_return = np.mean(stats["returns"])
            avg_length = np.mean(stats["episode_lengths"])
            
            print(f"  Avg Return: {avg_return:.3f} | Avg Length: {avg_length:.3f}")
            
            # Save results
            results["combinations"].append(params)
            results["returns"].append(avg_return)
            results["episode_lengths"].append(avg_length)
            
            # Update best params if needed
            if avg_return > results["best_return"]:
                results["best_return"] = avg_return
                results["best_params"] = params
            
            env.close()
        
        # Plot results
        self._plot_tuning_results(results, hyperparameter_grid)
        
        print(f"\n=== Tuning Complete ===")
        print(f"Best parameters: {results['best_params']}")
        print(f"Best average return: {results['best_return']:.3f}")
        
        return results
    
    def _calculate_grid_size(self, grid: Dict[str, List]) -> int:
        """Calculate the number of combinations in the grid"""
        size = 1
        for values in grid.values():
            size *= len(values)
        return size
    
    def _generate_combinations(self, grid: Dict[str, List]) -> List[Dict]:
        """Generate all combinations from the grid"""
        import itertools
        
        keys = list(grid.keys())
        values = list(grid.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _plot_tuning_results(self, results: Dict[str, Any], grid: Dict[str, List]):
        """Plot hyperparameter tuning results"""
        # Set top journal style
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'Arial',
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'lines.linewidth': 2,
            'axes.linewidth': 1,
            'grid.alpha': 0.7,
            'grid.linestyle': '--'
        })
        
        # For each hyperparameter, plot its effect on return
        params = list(grid.keys())
        num_params = len(params)
        
        if num_params > 0:
            plt.figure(figsize=(15, 5 * num_params))
            
            for i, param in enumerate(params):
                plt.subplot(num_params, 1, i+1)
                
                # Extract parameter values and corresponding returns
                param_values = []
                param_returns = []
                
                for combo, ret in zip(results["combinations"], results["returns"]):
                    param_values.append(combo[param])
                    param_returns.append(ret)
                
                # Sort by parameter value
                sorted_pairs = sorted(zip(param_values, param_returns))
                sorted_values, sorted_returns = zip(*sorted_pairs)
                
                plt.plot(sorted_values, sorted_returns, marker='o', linewidth=2, markersize=6)
                plt.title(f"Effect of {param} on Average Return")
                plt.xlabel(param)
                plt.ylabel("Average Return")
                plt.grid(True)
                
            plt.tight_layout()
            plt.savefig(os.path.join("figures", f"hyperparameter_tuning_{self.env_name}.png"), bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Hyperparameter tuning results saved as figures/hyperparameter_tuning_{self.env_name}.png")
        
        # Plot all combinations' returns
        plt.figure(figsize=(15, 8))
        
        # Create labels for each combination
        combo_labels = []
        for i, combo in enumerate(results["combinations"]):
            label = ", ".join([f"{k}={v}" for k, v in combo.items()])
            combo_labels.append(f"Comb {i+1}")
        
        plt.bar(combo_labels, results["returns"])
        plt.title("Hyperparameter Combination Performance")
        plt.xlabel("Combination")
        plt.ylabel("Average Return")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y')
        
        # Highlight best combination
        best_idx = results["returns"].index(results["best_return"])
        plt.bar(combo_labels[best_idx], results["best_return"], color='red', label="Best")
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("figures", f"hyperparameter_comparison_{self.env_name}.png"), bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Hyperparameter comparison saved as figures/hyperparameter_comparison_{self.env_name}.png")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuner for PPO")
    parser.add_argument("--env-name", type=str, default="myoElbowPose1D6MRandom-v0", help="Environment name")
    parser.add_argument("--total-timesteps", type=int, default=50000, help="Total timesteps per tuning run")
    args = parser.parse_args()
    
    # Define hyperparameter grid to search
    hyperparameter_grid = {
        "clip_range": [0.1, 0.2, 0.3],
        "lr": [1e-4, 3e-4, 1e-3],
        "gamma": [0.99, 0.995],
        "lambda_gae": [0.9, 0.95, 0.98]
    }
    
    # Create tuner and run tuning
    tuner = HyperparameterTuner(args.env_name)
    results = tuner.tune_hyperparameters(
        hyperparameter_grid=hyperparameter_grid,
        total_timesteps=args.total_timesteps
    )


if __name__ == "__main__":
    main()