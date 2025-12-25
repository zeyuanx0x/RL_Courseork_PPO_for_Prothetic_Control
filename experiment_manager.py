import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Callable, Any
from myosuite.utils import gym
from utils.env_utils import reset_compat, step_compat, safe_close
from ppo_agent import PPOAgent
from utils.render_utils import safe_render
from utils.plotting_utils import setup_plot_style, save_figure
from utils.eval_utils import run_policy


class ExperimentManager:
    """Experiment manager, responsible for environment inspection, policy testing, and result analysis"""
    
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.obs_space = self.env.observation_space
        self.act_space = self.env.action_space
        
    def env_inspection(self, render: bool = False) -> Dict[str, Any]:
        """Environment inspection, recording observation/action space information"""
        print(f"=== Environment Inspection: {self.env_name} ===")
        
        # Observation space information
        obs_info = {
            "shape": self.obs_space.shape,
            "low": self.obs_space.low,
            "high": self.obs_space.high,
            "is_box": hasattr(self.obs_space, 'low') and hasattr(self.obs_space, 'high')
        }
        
        # Action space information
        act_info = {
            "shape": self.act_space.shape,
            "low": self.act_space.low,
            "high": self.act_space.high,
            "is_box": hasattr(self.act_space, 'low') and hasattr(self.act_space, 'high')
        }
        
        print(f"Observation Space:")
        print(f"  Shape: {obs_info['shape']}")
        print(f"  Low: {obs_info['low'][:5]}...")
        print(f"  High: {obs_info['high'][:5]}...")
        
        print(f"\nAction Space:")
        print(f"  Shape: {act_info['shape']}")
        print(f"  Low: {act_info['low'][:5]}...")
        print(f"  High: {act_info['high'][:5]}...")
        
        # Run an episode, record key states
        obs, info = reset_compat(self.env)
        states = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "done": False
        }
        
        max_steps = getattr(getattr(self.env, "spec", None), "max_episode_steps", 1000)
        
        for step in range(max_steps):
            action = self.act_space.sample()
            # Use unified step compatibility function
            next_obs, reward, done, info = step_compat(self.env, action)
            
            states["obs"].append(obs)
            states["actions"].append(action)
            states["rewards"].append(reward)
            
            obs = next_obs
            
            if render:
                # Use unified safe render function
                safe_render(self.env)
            
            if done:
                states["done"] = True
                break
        
        print(f"\nEpisode Information:")
        print(f"  Steps: {len(states['obs'])}")
        print(f"  Total Reward: {sum(states['rewards'])}")
        print(f"  Done: {states['done']}")
        
        # Analyze if observation space contains goal error
        obs_array = np.array(states["obs"])
        has_goal_error = self._detect_goal_error(obs_array)
        print(f"  Has Goal Error Term: {has_goal_error}")
        
        # Plot key state curves
        self._plot_episode_states(states)
        
        return {
            "obs_info": obs_info,
            "act_info": act_info,
            "episode_states": states,
            "has_goal_error": has_goal_error
        }
    
    def _detect_goal_error(self, obs_array: np.ndarray) -> bool:
        """Detect if observation space contains goal error term"""
        # Simple detection: whether any dimension gradually decreases (approaches 0) during episode
        for i in range(obs_array.shape[1]):
            obs_dim = obs_array[:, i]
            # Calculate difference between final value and initial value
            diff = abs(obs_dim[-1] - obs_dim[0])
            # If difference is large and final value is close to 0, it might be goal error
            if diff > 0.5 and abs(obs_dim[-1]) < 0.1:
                return True
        return False
    
    def _plot_episode_states(self, states: Dict[str, List]) -> None:
        """Plot key state curves for the episode (top journal style)"""
        obs_array = np.array(states["obs"])
        actions_array = np.array(states["actions"])
        rewards = states["rewards"]
        
        # Set plot style using unified function
        setup_plot_style()
        
        fig = plt.figure(figsize=(12, 10))
        
        # Plot the first 5 observation dimensions
        plt.subplot(3, 1, 1)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i in range(min(5, obs_array.shape[1])):
            plt.plot(obs_array[:, i], label=f"Obs Dim {i+1}", color=colors[i % len(colors)])
        plt.title("(a) Observation Dimensions")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend(ncol=min(5, obs_array.shape[1]), loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Plot the first 5 action dimensions
        plt.subplot(3, 1, 2)
        for i in range(min(5, actions_array.shape[1])):
            plt.plot(actions_array[:, i], label=f"Act Dim {i+1}", color=colors[i % len(colors)])
        plt.title("(b) Action Dimensions")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend(ncol=min(5, actions_array.shape[1]), loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Plot rewards
        plt.subplot(3, 1, 3)
        plt.plot(rewards, color='#8c564b')
        plt.title("(c) Reward per Step")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure using unified function
        save_figure(fig, f"env_inspection_{self.env_name}.png", ".")
    
    def zero_policy(self, obs: np.ndarray) -> np.ndarray:
        """Zero policy: all actions are zero"""
        return np.zeros(self.act_space.shape)
    
    def random_policy(self, obs: np.ndarray) -> np.ndarray:
        """Random policy"""
        return self.act_space.sample()
    
    def simple_heuristic(self, obs: np.ndarray) -> np.ndarray:
        """Simple heuristic policy: apply a small torque along the error direction"""
        # Assume the first few dimensions of observation are joint angle errors
        error = obs[:min(3, len(obs))]
        action = -0.1 * error  # Simple proportional controller
        # Extend action to match the action space shape
        full_action = np.zeros(self.act_space.shape)
        full_action[:len(action)] = action
        # Clip to action space bounds
        return np.clip(full_action, self.act_space.low, self.act_space.high)
    
    def run_policy(self, policy: Callable, episodes: int = 20, render: bool = False) -> Dict[str, List]:
        """Run policy and collect data (using unified eval_utils.run_policy)"""
        return run_policy(self.env, policy, episodes, render, safe_render)
    
    def sanity_check(self, episodes: int = 20, render: bool = False) -> Dict[str, Dict[str, List]]:
        """Three-policy sanity check"""
        print(f"=== Sanity Check: {self.env_name} ===")
        
        policies = {
            "zero": self.zero_policy,
            "random": self.random_policy,
            "heuristic": self.simple_heuristic
        }
        
        results = {}
        
        for policy_name, policy_fn in policies.items():
            print(f"\nRunning {policy_name} policy...")
            policy_results = self.run_policy(policy_fn, episodes, render)
            results[policy_name] = policy_results
            
            # Calculate statistical information
            returns = policy_results["returns"]
            lengths = policy_results["episode_lengths"]
            
            print(f"  Mean Return: {np.mean(returns):.3f} ± {np.std(returns):.3f}")
            print(f"  Mean Episode Length: {np.mean(lengths):.3f} ± {np.std(lengths):.3f}")
            print(f"  Min Return: {np.min(returns):.3f}")
            print(f"  Max Return: {np.max(returns):.3f}")
        
        # Plot result comparison
        self._plot_sanity_check(results)
        
        return results
    
    def _plot_sanity_check(self, results: Dict[str, Dict[str, List]]) -> None:
        """Plot sanity check result comparison (top journal style)"""
        # Set top journal style parameters
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
            'xtick.major.width': 1,
            'ytick.major.width': 1,
            'xtick.major.size': 5,
            'ytick.major.size': 5,
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'legend.edgecolor': 'black'
        })
        
        plt.figure(figsize=(15, 12))
        
        policies = list(results.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Plot return distribution (box plot)
        plt.subplot(2, 2, 1)
        returns = [results[policy]["returns"] for policy in policies]
        box_plot = plt.boxplot(returns, labels=policies, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors[:len(policies)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for whisker in box_plot['whiskers']:
            whisker.set(color='black', linewidth=1.5)
        for cap in box_plot['caps']:
            cap.set(color='black', linewidth=1.5)
        for median in box_plot['medians']:
            median.set(color='black', linewidth=2)
        plt.title("(a) Return Distribution")
        plt.ylabel("Return")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Plot returns per episode
        plt.subplot(2, 2, 2)
        for i, (policy_name, policy_results) in enumerate(results.items()):
            plt.plot(policy_results["returns"], label=policy_name, color=colors[i])
        plt.title("(b) Returns per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.legend(ncol=len(policies), loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Plot average reward
        plt.subplot(2, 2, 3)
        for i, (policy_name, policy_results) in enumerate(results.items()):
            avg_rewards = [np.mean(rews) for rews in policy_results["all_rewards"]]
            plt.plot(avg_rewards, label=policy_name, color=colors[i])
        plt.title("(c) Average Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.legend(ncol=len(policies), loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Plot average action magnitude
        plt.subplot(2, 2, 4)
        for i, (policy_name, policy_results) in enumerate(results.items()):
            avg_action_mags = [np.mean(np.abs(acts)) for acts in policy_results["all_actions"]]
            plt.plot(avg_action_mags, label=policy_name, color=colors[i])
        plt.title("(d) Average Action Magnitude per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Average Action Magnitude")
        plt.legend(ncol=len(policies), loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Ensure results folder exists
        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/sanity_check_{self.env_name}.png", bbox_inches='tight', dpi=300, format='png')
        plt.close()
        print(f"Sanity check plot saved as results/sanity_check_{self.env_name}.png")
    
    def close(self):
        """Close the environment using unified safe_close function"""
        safe_close(self.env)


def main():
    """Main function, used to test the experiment manager"""
    env_name = "myoElbowPose1D6MRandom-v0"
    
    # Create experiment manager
    manager = ExperimentManager(env_name)
    
    # Environment inspection
    manager.env_inspection(render=False)
    
    # Three-policy sanity check
    manager.sanity_check(episodes=20, render=False)
    
    manager.close()


if __name__ == "__main__":
    main()
