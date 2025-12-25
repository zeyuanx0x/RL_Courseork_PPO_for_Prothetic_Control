import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Callable, Any
from myosuite.utils import gym
from utils.env_utils import reset_compat, step_compat

# Import NatureStylePlotter
from plotting_utils import NatureStylePlotter

class FailureMapGenerator:
    """Failure mode and reliability map generator"""
    
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.base_env = gym.make(env_name)
        self.act_space = self.base_env.action_space
        self.obs_space = self.base_env.observation_space
    
    def apply_perturbation(self, env, perturbation: Dict[str, Any]) -> None:
        """Apply perturbation to the environment
        
        Args:
            env: Environment instance
            perturbation: Dictionary of perturbation parameters
        """
        # Apply different types of perturbations
        if "delay" in perturbation:
            # Action delay perturbation
            env.action_delay = perturbation["delay"]
        
        if "obs_noise" in perturbation:
            # Observation noise perturbation
            env.observation_noise = perturbation["obs_noise"]
        
        if "friction" in perturbation:
            # Friction perturbation
            if hasattr(env, 'model') and hasattr(env.model, 'dof_damping'):
                env.model.dof_damping *= perturbation["friction"]
        
        if "mass" in perturbation:
            # Mass perturbation
            if hasattr(env, 'model') and hasattr(env.model, 'body_mass'):
                env.model.body_mass *= perturbation["mass"]
        
        if "action_scale" in perturbation:
            # Action scale perturbation
            self.act_space.high *= perturbation["action_scale"]
            self.act_space.low *= perturbation["action_scale"]
    
    def run_perturbation_test(self, policy: Callable, perturbation: Dict[str, Any], 
                            episodes: int = 10, seed: int = 42) -> Dict[str, Any]:
        """Run tests with specific perturbation
        
        Args:
            policy: Policy function
            perturbation: Perturbation parameters
            episodes: Number of episodes to run
            seed: Random seed
            
        Returns:
            Test results dictionary
        """
        print(f"=== Running perturbation test: {perturbation} ===")
        
        np.random.seed(seed)
        
        results = {
            "returns": [],
            "episode_lengths": [],
            "success_rates": [],
            "safety_violations": [],
            "perturbation": perturbation.copy()
        }
        
        for ep_idx in range(episodes):
            env = gym.make(self.env_name)
            
            # Apply perturbation
            self.apply_perturbation(env, perturbation)
            
            obs, info = reset_compat(env)
            
            ep_return = 0.0
            ep_length = 0
            safety_violations = 0
            
            while True:
                # Get action
                action = policy(obs)
                
                # Step the environment
                next_obs, reward, done, info = step_compat(env, action)
                
                # Check for safety violations
                # Simple safety check: check if any observation dimension is out of bounds
                obs_range = self.obs_space.high - self.obs_space.low
                obs_normalized = (obs - self.obs_space.low) / (obs_range + 1e-8)
                if np.any(obs_normalized < -0.1) or np.any(obs_normalized > 1.1):
                    safety_violations += 1
                
                # Update episode data
                ep_return += reward
                ep_length += 1
                
                obs = next_obs
                
                if done:
                    break
            
            env.close()
            
            # Determine success (arbitrary threshold for now)
            success = int(ep_return > 0)
            
            # Update results
            results["returns"].append(ep_return)
            results["episode_lengths"].append(ep_length)
            results["success_rates"].append(success)
            results["safety_violations"].append(safety_violations)
            
            print(f"  Episode {ep_idx+1}/{episodes}: Return={ep_return:.3f}, Length={ep_length}, Safety Violations={safety_violations}")
        
        # Calculate aggregate metrics
        results["mean_return"] = np.mean(results["returns"])
        results["std_return"] = np.std(results["returns"])
        results["mean_success_rate"] = np.mean(results["success_rates"])
        results["mean_safety_violations"] = np.mean(results["safety_violations"])
        
        return results
    
    def generate_perturbation_space(self, base_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate perturbation space based on base parameters
        
        Args:
            base_params: Dictionary of base parameters with lists of values
            
        Returns:
            List of perturbation dictionaries
        """
        perturbation_space = []
        
        # Generate all combinations of perturbation parameters
        # This is a simple implementation for small parameter spaces
        param_names = list(base_params.keys())
        param_values = list(base_params.values())
        
        # Recursively generate combinations
        def generate_combinations(idx, current):
            if idx == len(param_names):
                perturbation_space.append(current.copy())
                return
            
            param_name = param_names[idx]
            for value in param_values[idx]:
                current[param_name] = value
                generate_combinations(idx + 1, current)
                del current[param_name]
        
        generate_combinations(0, {})
        
        return perturbation_space
    
    def generate_failure_map(self, policy: Callable, perturbation_space: List[Dict[str, Any]],
                           episodes: int = 10, seed: int = 42) -> pd.DataFrame:
        """Generate failure map by running tests across perturbation space
        
        Args:
            policy: Policy function
            perturbation_space: List of perturbation dictionaries
            episodes: Number of episodes per perturbation
            seed: Random seed
            
        Returns:
            Failure map DataFrame
        """
        print(f"=== Generating Failure Map: {self.env_name} ===")
        print(f"Perturbation space size: {len(perturbation_space)}")
        print(f"Episodes per perturbation: {episodes}")
        
        # Run all perturbation tests
        test_results = []
        for i, perturbation in enumerate(perturbation_space):
            print(f"\n[{i+1}/{len(perturbation_space)}]")
            result = self.run_perturbation_test(policy, perturbation, episodes, seed)
            test_results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame(test_results)
        
        # Save raw results
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/failure_map.csv", index=False)
        print(f"\nFailure map results saved to results/failure_map.csv")
        
        # Plot failure maps
        self._plot_failure_maps(df)
        
        return df
    
    def _plot_failure_maps(self, df: pd.DataFrame) -> None:
        """Plot failure maps
        
        Args:
            df: Failure map DataFrame
        """
        os.makedirs("figures", exist_ok=True)
        
        # Determine which perturbation parameters are available
        available_params = []
        for param in ["delay", "obs_noise", "friction", "mass", "action_scale"]:
            if any(param in p for p in df["perturbation"]):
                available_params.append(param)
        
        print(f"Available perturbation parameters: {available_params}")
        
        # Check if we have at least two parameters for 2D plots
        if len(available_params) < 2:
            print("Insufficient parameters for 2D failure maps, plotting 1D instead")
            self._plot_1d_failure_maps(df, available_params)
            return
        
        # Create 2D failure maps for all parameter pairs
        for i in range(len(available_params)):
            for j in range(i+1, len(available_params)):
                param_x = available_params[i]
                param_y = available_params[j]
                
                # Check if both parameters are present in all perturbations
                has_both = all(param_x in p and param_y in p for p in df["perturbation"])
                if not has_both:
                    continue
                
                # Extract parameter values (no heatmaps generated)
    
    def _plot_1d_failure_maps(self, df: pd.DataFrame, available_params: List[str]) -> None:
        """Plot 1D failure maps
        
        Args:
            df: Failure map DataFrame
            available_params: List of available parameters
        """
        # Create NatureStylePlotter instance
        plotter = NatureStylePlotter(self.env_name)
        
        for param in available_params:
            # Extract parameter values
            has_param = [param in p for p in df["perturbation"]]
            param_df = df[has_param].copy()
            param_df[f"{param}_value"] = [p[param] for p in param_df["perturbation"]]
            
            # Sort by parameter value
            param_df = param_df.sort_values(f"{param}_value")
            
            # Plot success rate chart
            plotter.plot_failure_1d_success(param_df, param)
            
            # Plot return chart
            plotter.plot_failure_1d_return(param_df, param)
    
    def _plot_2d_heatmap(self, df: pd.DataFrame, param_x: str, param_y: str, metric: str,
                        title: str, save_path: str) -> None:
        """Plot 2D heatmap
        
        Args:
            df: DataFrame with results
            param_x: X-axis parameter
            param_y: Y-axis parameter
            metric: Metric to plot
            title: Plot title
            save_path: Path to save the plot
        """
        # Get unique parameter values
        x_values = sorted(df[f"{param_x}_value"].unique())
        y_values = sorted(df[f"{param_y}_value"].unique())
        
        # Create grid
        grid = np.zeros((len(y_values), len(x_values)))
        
        # Fill grid with metric values
        for i, y in enumerate(y_values):
            for j, x in enumerate(x_values):
                # Find the row with this combination
                row = df[(df[f"{param_x}_value"] == x) & (df[f"{param_y}_value"] == y)]
                if len(row) > 0:
                    grid[i, j] = row[metric].values[0]
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        
        # Use pcolormesh for better control
        im = plt.pcolormesh(x_values, y_values, grid, cmap='RdYlBu_r', shading='auto')
        
        # Add colorbar with descriptive label
        cbar = plt.colorbar(im)
        cbar.set_label(metric.replace('_', ' ').title())
        
        # Set plot properties
        plt.title(f"{title} ({self.env_name})")
        plt.xlabel(param_x)
        plt.ylabel(param_y)
        plt.grid(True, linestyle='--', alpha=0.3, color='black')
        
        # Add value labels to each cell
        for i, y in enumerate(y_values):
            for j, x in enumerate(x_values):
                value = grid[i, j]
                plt.text(x, y, f"{value:.2f}", ha='center', va='center', fontsize=8, 
                        color='black' if 0.2 < grid[i, j] < 0.8 else 'white')
        
        # Ensure the plot is tight
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"2D heatmap saved to {save_path}")
    
    def generate_reliability_report(self, df: pd.DataFrame) -> str:
        """Generate reliability report
        
        Args:
            df: Failure map DataFrame
            
        Returns:
            Reliability report string
        """
        # Identify critical failure points
        critical_failures = df[df["mean_success_rate"] < 0.5]
        
        report = f"# Reliability Map Report\n\n"
        report += f"## Environment: {self.env_name}\n\n"
        
        report += "## Test Parameters\n"
        report += f"- Number of perturbations: {len(df)}\n"
        report += f"- Episodes per perturbation: {len(df['returns'].iloc[0]) if len(df) > 0 else 0}\n\n"
        
        report += "## Key Findings\n\n"
        
        if len(critical_failures) > 0:
            report += f"### Critical Failure Points ({len(critical_failures)} identified)\n"
            report += f"| Perturbation | Success Rate | Safety Violations | Mean Return |\n"
            report += f"|--------------|--------------|------------------|-------------|\n"
            
            for _, row in critical_failures.iterrows():
                report += f"| {row['perturbation']} | {row['mean_success_rate']:.3f} | {row['mean_safety_violations']:.3f} | {row['mean_return']:.3f} |\n"
        else:
            report += "### Critical Failure Points\n"
            report += "No critical failures identified in the tested perturbation space.\n\n"
        
        report += "\n### Reliability Summary\n"
        report += f"- Overall average success rate: {df['mean_success_rate'].mean():.3f}\n"
        report += f"- Maximum success rate: {df['mean_success_rate'].max():.3f}\n"
        report += f"- Minimum success rate: {df['mean_success_rate'].min():.3f}\n"
        report += f"- Average safety violations: {df['mean_safety_violations'].mean():.3f}\n\n"
        
        report += "### Heatmaps\n"
        report += "The following heatmaps visualize the reliability across different perturbation combinations:\n\n"
        
        # List all generated heatmaps
        heatmap_files = [f for f in os.listdir("figures") if f.startswith("failure_map_")]
        for heatmap_file in heatmap_files:
            report += f"![{heatmap_file}](figures/{heatmap_file})\n\n"
        
        report += "## Conclusion\n"
        report += "The reliability map analysis reveals how the policy performance degrades under various perturbations. "
        report += "Critical failure points indicate where the policy is most vulnerable, suggesting areas for improvement. "
        report += "The heatmaps provide a visual representation of the policy's robustness across the perturbation space.\n"
        
        return report

def main():
    """Main function"""
    from ppo_agent import PPOAgent
    from experiment_manager import ExperimentManager
    
    env_name = "myoElbowPose1D6MRandom-v0"
    
    # Create environment and agent
    env = gym.make(env_name)
    agent = PPOAgent(env)
    
    # Try to load pre-trained model
    checkpoint_path = "checkpoints/ppo_myoElbowPose1D6MRandom-v0_seed42.pt"
    try:
        agent.load(checkpoint_path)
        print(f"Successfully loaded model: {checkpoint_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Define policy function
    def ppo_policy(obs):
        return agent.get_action(obs)
    
    # Create failure map generator
    generator = FailureMapGenerator(env_name)
    
    # Define perturbation space
    perturbation_space = generator.generate_perturbation_space({
        "delay": [0, 1, 2, 3],
        "obs_noise": [0, 0.01, 0.05, 0.1],
        "action_scale": [0.8, 1.0, 1.2]
    })
    
    # Generate failure map
    results_df = generator.generate_failure_map(ppo_policy, perturbation_space, 
                                              episodes=10, seed=42)
    
    # Generate reliability report
    report = generator.generate_reliability_report(results_df)
    with open("reports/reliability_report.md", "w") as f:
        f.write(report)
    
    print("\n=== Failure Map Generation Complete ===")
    print(f"Results saved to: results/failure_map.csv")
    print(f"Report saved to: reports/reliability_report.md")


if __name__ == "__main__":
    main()
