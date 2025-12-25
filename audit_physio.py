import numpy as np
import os
import pandas as pd
from typing import Dict, List, Callable, Any
from myosuite.utils import gym
from utils.env_utils import reset_compat, step_compat

# Import NatureStylePlotter
from plotting_utils import NatureStylePlotter

class PhysiologyAuditor:
    """Physiology/Compensation Auditor"""
    
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.base_env = gym.make(env_name)
        
    def audit_episode(self, policy: Callable, episode_idx: int, algo: str, seed: int) -> Dict[str, Any]:
        """Audit a single episode
        
        Args:
            policy: Policy function
            episode_idx: Episode index
            algo: Algorithm name
            seed: Random seed
            
        Returns:
            Audit results dictionary
        """
        env = gym.make(self.env_name)
        obs, info = reset_compat(env)
        
        # Collect data
        episode_data = {
            "episode_idx": episode_idx,
            "algo": algo,
            "seed": seed,
            "actions": [],
            "rewards": [],
            "obs": [],
            "info": [],
            "muscle_activations": [],
        }
        
        # Run episode
        while True:
            action = policy(obs)
            next_obs, reward, done, info = step_compat(env, action)
            
            # Record data
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            episode_data["obs"].append(obs)
            episode_data["info"].append(info)
            
            # Try to get muscle activation data
            if "muscle_activation" in info:
                episode_data["muscle_activations"].append(info["muscle_activation"])
            elif "muscle_activations" in info:
                episode_data["muscle_activations"].append(info["muscle_activations"])
            
            obs = next_obs
            if done:
                break
        
        env.close()
        
        # Calculate audit metrics
        audit_results = self._calculate_audit_metrics(episode_data)
        return audit_results
    
    def _calculate_audit_metrics(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate audit metrics
        
        Args:
            episode_data: Episode data
            
        Returns:
            Audit metrics dictionary
        """
        actions = np.array(episode_data["actions"])
        obs = np.array(episode_data["obs"])
        muscle_activations = np.array(episode_data["muscle_activations"])
        
        results = {
            "episode_idx": episode_data["episode_idx"],
            "algo": episode_data["algo"],
            "seed": episode_data["seed"],
        }
        
        # A. Muscle/force proxy
        if muscle_activations.size > 0:
            # Use muscle activation as force proxy
            results["muscle_activation_rms"] = np.sqrt(np.mean(muscle_activations**2))
            results["muscle_activation_peak"] = np.max(muscle_activations)
            results["muscle_activation_mean"] = np.mean(muscle_activations)
        else:
            # Fallback: Use action squared integral as force proxy
            results["action_force_proxy"] = np.sum(actions**2)  # ∑ action^2 dt
            results["action_force_proxy_mean"] = np.mean(actions**2)
            results["action_force_proxy_peak"] = np.max(actions**2)
        
        # B. Compensation metrics
        if actions.shape[1] > 1:
            # Calculate channel burden
            channel_load = np.sum(np.abs(actions), axis=0)
            total_load = np.sum(channel_load)
            if total_load > 0:
                # Assume first channel is the target channel
                target_channel_load = channel_load[0]
                non_target_channel_load = total_load - target_channel_load
                results["target_channel_burden"] = target_channel_load / total_load
                results["non_target_channel_burden"] = non_target_channel_load / total_load
                results["burden_shift"] = non_target_channel_load - target_channel_load
        
        # C. Smoothness / comfort proxy
        if len(actions) > 1:
            # First-order difference (L1)
            delta_actions = np.diff(actions, axis=0)
            results["action_smoothness_l1"] = np.mean(np.abs(delta_actions))
            results["action_smoothness_l2"] = np.mean(delta_actions**2)
            
            # Second-order difference (jerk)
            if len(delta_actions) > 1:
                delta2_actions = np.diff(delta_actions, axis=0)
                results["action_jerk"] = np.mean(np.abs(delta2_actions))
        
        # D. Safety boundaries
        # Joint angle violations (simplified: assume first few observation dimensions are joint angles)
        if len(obs) > 0 and len(obs[0]) > 0:
            # Assume joint angle range is [-pi, pi]
            joint_angles = obs[:, :min(3, obs.shape[1])]
            angle_violations = np.sum((joint_angles < -np.pi) | (joint_angles > np.pi))
            results["joint_angle_violations"] = angle_violations
        else:
            results["joint_angle_violations"] = 0
        
        # Joint velocity violations (assume joint velocities are in observations)
        if len(obs) > 0 and len(obs[0]) > 3:
            joint_velocities = obs[:, 3:min(6, obs.shape[1])]
            # Assume angular velocity limit is 10 rad/s
            velocity_violations = np.sum(np.abs(joint_velocities) > 10.0)
            results["joint_velocity_violations"] = velocity_violations
        else:
            results["joint_velocity_violations"] = 0
        
        return results
    
    def run_audit(self, policy: Callable, episodes: int = 10, algo: str = "PPO", seed: int = 42) -> pd.DataFrame:
        """Run audit
        
        Args:
            policy: Policy function
            episodes: Number of episodes
            algo: Algorithm name
            seed: Random seed
            
        Returns:
            Audit results DataFrame
        """
        print(f"=== Physiology Audit: {self.env_name} ===")
        print(f"Algorithm: {algo}, Seed: {seed}, Episodes: {episodes}")
        
        results = []
        for ep in range(episodes):
            audit_result = self.audit_episode(policy, ep, algo, seed)
            results.append(audit_result)
            print(f"  Episode {ep+1} completed")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Ensure results directories exist
        os.makedirs("results", exist_ok=True)
        os.makedirs("figures", exist_ok=True)
        
        # Save audit results
        df.to_csv("results/physio_audit.csv", index=False)
        print(f"\nAudit results saved to results/physio_audit.csv")
        
        # Generate statistical summary
        self._generate_statistical_summary(df)
        
        # Generate figures
        self._plot_audit_results(df)
        
        return df
    
    def _generate_statistical_summary(self, df: pd.DataFrame):
        """Generate statistical summary
        
        Args:
            df: Audit results DataFrame
        """
        # Only aggregate numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate statistical summary
        summary = numeric_df.agg(["mean", "median", "std", "min", "max"])
        
        # Add 90% and 95% percentiles
        summary.loc["quantile_90"] = numeric_df.quantile(0.9)
        summary.loc["quantile_95"] = numeric_df.quantile(0.95)
        
        # Save statistical summary
        summary.to_csv("results/physio_audit_summary.csv")
        print(f"Statistical summary saved to results/physio_audit_summary.csv")
    
    def _plot_audit_results(self, df: pd.DataFrame):
        """Generate audit results figures
        
        Args:
            df: Audit results DataFrame
        """
        # Create NatureStylePlotter instance
        plotter = NatureStylePlotter(self.env_name)
        
        # Generate force proxy figure
        if "action_force_proxy" in df.columns:
            plotter.plot_physio_force_proxy(df)
        
        # Generate smoothness figure
        if "action_smoothness_l1" in df.columns:
            plotter.plot_physio_smoothness(df)
        
        # Generate jerk figure
        if "action_jerk" in df.columns:
            plotter.plot_physio_jerk(df)
        
        # Generate safety boundary figure
        plotter.plot_physio_safety(df)


def main():
    """Main function"""
    # Import necessary modules
    from ppo_agent import PPOAgent
    from experiment_manager import ExperimentManager
    
    # Set environment name
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
    
    # Create auditor
    auditor = PhysiologyAuditor(env_name)
    
    # Define policy function
    def ppo_policy(obs):
        return agent.get_action(obs)
    
    # Run audit
    auditor.run_audit(ppo_policy, episodes=10, algo="PPO", seed=42)
    
    print("\n=== Physiology Audit Completed ===")


if __name__ == "__main__":
    main()
