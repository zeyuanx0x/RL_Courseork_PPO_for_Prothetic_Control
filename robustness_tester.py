import numpy as np
import os
from typing import Dict, List, Callable, Any
from myosuite.utils import gym
from utils.env_utils import make_env, reset_compat, step_compat, safe_close
from utils.render_utils import safe_render
from utils.plotting_utils import setup_plot_style, save_figure

# Try to import matplotlib, if it fails, we'll skip plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available, plotting will be skipped")
    MATPLOTLIB_AVAILABLE = False


class RobustnessTester:
    """Robustness tester, responsible for domain randomization, perturbation testing, and generalization evaluation"""
    
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.base_env = make_env(env_name)
    
    def domain_randomization_test(
        self,
        policy: Callable,
        randomization_params: Dict[str, List[float]],
        episodes: int = 10
    ) -> Dict[str, List]:
        """Domain randomization test
        
        Args:
            policy: Policy function to test
            randomization_params: Randomization parameters in format {param_name: [min_val, max_val]}
            episodes: Number of test episodes per randomization setting
            
        Returns:
            Test results containing returns and success rates under different randomization settings
        """
        print(f"=== Domain Randomization Test: {self.env_name} ===")
        
        results = {
            "param_settings": [],
            "returns": [],
            "episode_lengths": [],
            "success_rates": []
        }
        
        # Generate randomization parameter combinations
        for i in range(5):  # Generate 5 randomization settings
            # Randomly generate parameter values
            param_setting = {}
            for param_name, (min_val, max_val) in randomization_params.items():
                param_setting[param_name] = np.random.uniform(min_val, max_val)
            
            print(f"\nTesting with param setting {i+1}: {param_setting}")
            
            # Create environment with randomization parameters using unified make_env function
            env = make_env(self.env_name, **param_setting)
            
            # Run tests
            ep_returns = []
            ep_lengths = []
            successes = 0
            
            for ep in range(episodes):
                # Use unified reset compatibility function
                obs, info = reset_compat(env)
                ep_return = 0.0
                ep_steps = 0
                
                while True:
                    action = policy(obs)
                    # Use unified step compatibility function
                    next_obs, reward, done, info = step_compat(env, action)
                    
                    ep_return += reward
                    ep_steps += 1
                    
                    obs = next_obs
                    
                    if done:
                        # Check if successful (based on environment info)
                        if "success" in info and info["success"]:
                            successes += 1
                        break
                
                ep_returns.append(ep_return)
                ep_lengths.append(ep_steps)
            
            # Calculate statistical information
            avg_return = np.mean(ep_returns)
            avg_length = np.mean(ep_lengths)
            success_rate = successes / episodes
            
            print(f"  Avg Return: {avg_return:.3f} | Avg Length: {avg_length:.3f} | Success Rate: {success_rate:.3f}")
            
            # Save results
            results["param_settings"].append(param_setting)
            results["returns"].append(ep_returns)
            results["episode_lengths"].append(ep_lengths)
            results["success_rates"].append(success_rate)
            
            # Use unified safe close function
            safe_close(env)
        
        # Plot results
        self._plot_domain_randomization_results(results)
        
        return results
    
    def perturbation_test(
        self,
        policy: Callable,
        perturbation_type: str = "force",
        perturbation_strength: float = 0.1,
        episodes: int = 10
    ) -> Dict[str, List]:
        """Perturbation test
        
        Args:
            policy: Policy function to test
            perturbation_type: Perturbation type, options: force, noise, delay
            perturbation_strength: Perturbation strength
            episodes: Number of test episodes
            
        Returns:
            Test results
        """
        print(f"=== Perturbation Test: {self.env_name} ===")
        print(f"  Perturbation Type: {perturbation_type}")
        print(f"  Perturbation Strength: {perturbation_strength}")
        
        results = {
            "returns": [],
            "episode_lengths": [],
            "recovery_times": [],  # Time to recover to stable state
            "max_deviations": []   # Maximum deviation
        }
        
        env = gym.make(self.env_name)
        
        for ep in range(episodes):
            obs, info = env.reset()
            ep_return = 0.0
            ep_steps = 0
            
            # Record initial state for deviation calculation
            initial_obs = obs.copy()
            max_deviation = 0.0
            recovery_time = -1
            
            # Action history (for delay perturbation)
            action_history = []
            
            while True:
                # Apply perturbation
                if perturbation_type == "delay" and ep_steps > 0:
                    # Delay action execution
                    if len(action_history) >= perturbation_strength:
                        action = action_history[-int(perturbation_strength)]
                    else:
                        action = policy(obs)
                else:
                    action = policy(obs)
                
                # Execute action
                step_result = env.step(action)
                if len(step_result) == 5:
                    # Gymnasium: next_obs, reward, terminated, truncated, info
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    # Gym: next_obs, reward, done, info
                    next_obs, reward, done, info = step_result
                
                # Apply external force perturbation
                if perturbation_type == "force" and ep_steps % 50 == 10:  # Apply perturbation at step 10 every 50 steps
                    # Assume action space is muscle activation, we directly modify observation or action
                    # Simplified processing: directly add perturbation to observation
                    next_obs = next_obs + np.random.normal(0, perturbation_strength, size=next_obs.shape)
                
                # Apply observation noise
                elif perturbation_type == "noise":
                    next_obs = next_obs + np.random.normal(0, perturbation_strength, size=next_obs.shape)
                
                # Calculate deviation
                deviation = np.linalg.norm(next_obs - initial_obs)
                max_deviation = max(max_deviation, deviation)
                
                # Check if recovered (deviation less than 1.2 times initial deviation)
                if recovery_time == -1 and deviation < 1.2 * np.linalg.norm(initial_obs):
                    recovery_time = ep_steps
                
                ep_return += reward
                ep_steps += 1
                
                # Save action history
                action_history.append(action)
                
                obs = next_obs
                
                if done:
                    break
            
            # Save results
            results["returns"].append(ep_return)
            results["episode_lengths"].append(ep_steps)
            results["recovery_times"].append(recovery_time if recovery_time != -1 else ep_steps)
            results["max_deviations"].append(max_deviation)
            
            print(f"  Episode {ep+1}: Return={ep_return:.3f}, Length={ep_steps}, Recovery={recovery_time if recovery_time != -1 else 'N/A'}, Max Dev={max_deviation:.3f}")
        
        env.close()
        
        # Plot results
        self._plot_perturbation_results(results, perturbation_type)
        
        return results
    
    def multi_task_generalization(
        self,
        policy: Callable,
        target_envs: List[str],
        episodes: int = 10
    ) -> Dict[str, Dict[str, List]]:
        """Multi-task generalization test
        
        Args:
            policy: Policy function to test
            target_envs: List of target environments
            episodes: Number of test episodes per environment
            
        Returns:
            Test results containing returns and success rates for each target environment
        """
        print(f"=== Multi-Task Generalization Test ===")
        
        results = {}
        
        for target_env in target_envs:
            print(f"\nTesting on env: {target_env}")
            
            env = gym.make(target_env)
            
            ep_returns = []
            ep_lengths = []
            successes = 0
            
            for ep in range(episodes):
                obs, info = env.reset()
                ep_return = 0.0
                ep_steps = 0
                
                while True:
                    action = policy(obs)
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        # Gymnasium: next_obs, reward, terminated, truncated, info
                        next_obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        # Gym: next_obs, reward, done, info
                        next_obs, reward, done, info = step_result
                    
                    ep_return += reward
                    ep_steps += 1
                    
                    obs = next_obs
                    
                    if done:
                        if "success" in info and info["success"]:
                            successes += 1
                        break
                
                ep_returns.append(ep_return)
                ep_lengths.append(ep_steps)
            
            # Calculate statistical information
            avg_return = np.mean(ep_returns)
            avg_length = np.mean(ep_lengths)
            success_rate = successes / episodes
            
            print(f"  Avg Return: {avg_return:.3f} | Avg Length: {avg_length:.3f} | Success Rate: {success_rate:.3f}")
            
            # Save results
            results[target_env] = {
                "returns": ep_returns,
                "episode_lengths": ep_lengths,
                "success_rate": success_rate
            }
            
            env.close()
        
        # Plot results
        self._plot_multi_task_results(results)
        
        return results
    
    def _plot_domain_randomization_results(self, results: Dict[str, List]):
        """Plot domain randomization test results"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        plt.figure(figsize=(15, 5))
        
        # Plot return box plots under different parameter settings
        returns = results["returns"]
        plt.boxplot(returns, labels=[f"Set {i+1}" for i in range(len(returns))])
        plt.title("Returns under Different Domain Randomization Settings")
        plt.ylabel("Return")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join("figures", f"domain_randomization_{self.env_name}.png"), dpi=300)
        plt.close()
        print(f"Domain randomization results saved as figures/domain_randomization_{self.env_name}.png")
    
    def _plot_perturbation_results(self, results: Dict[str, List], perturbation_type: str):
        """Plot perturbation test results"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        plt.figure(figsize=(15, 10))
        
        # Plot returns
        plt.subplot(2, 2, 1)
        plt.plot(results["returns"])
        plt.title("Returns")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.grid(True)
        
        # Plot recovery times
        plt.subplot(2, 2, 2)
        plt.plot(results["recovery_times"])
        plt.title("Recovery Times")
        plt.xlabel("Episode")
        plt.ylabel("Recovery Time (steps)")
        plt.grid(True)
        
        # Plot maximum deviations
        plt.subplot(2, 2, 3)
        plt.plot(results["max_deviations"])
        plt.title("Max Deviations")
        plt.xlabel("Episode")
        plt.ylabel("Max Deviation")
        plt.grid(True)
        
        # Plot statistical summary
        plt.subplot(2, 2, 4)
        stats = [
            np.mean(results["returns"]),
            np.mean(results["recovery_times"]),
            np.mean(results["max_deviations"])
        ]
        labels = ["Avg Return", "Avg Recovery Time", "Avg Max Deviation"]
        plt.bar(labels, stats)
        plt.title("Perturbation Test Summary")
        plt.ylabel("Value")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join("figures", f"perturbation_{perturbation_type}_{self.env_name}.png"))
        plt.close()
        print(f"Perturbation results saved as figures/perturbation_{perturbation_type}_{self.env_name}.png")
    
    def _plot_multi_task_results(self, results: Dict[str, Dict[str, List]]):
        """Plot multi-task generalization results"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        plt.figure(figsize=(15, 5))
        
        # Plot return box plots across different environments
        envs = list(results.keys())
        returns = [results[env]["returns"] for env in envs]
        plt.boxplot(returns, labels=envs)
        plt.title("Returns across Different Environments")
        plt.ylabel("Return")
        plt.xticks(rotation=45)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join("figures", f"multi_task_generalization_{self.env_name}.png"), dpi=300)
        plt.close()
        print(f"Multi-task generalization results saved as figures/multi_task_generalization_{self.env_name}.png")
    
    def close(self):
        """Close the environment"""
        self.base_env.close()
