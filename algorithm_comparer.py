import numpy as np
import os
from typing import Dict, List, Callable, Any, Optional
from myosuite.utils import gym

# Try to import matplotlib, if it fails, we'll skip plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available, plotting will be skipped")
    MATPLOTLIB_AVAILABLE = False


class AlgorithmComparer:
    """Algorithm comparator, supports comparing performance of different algorithms"""
    
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.base_env = gym.make(env_name)
    
    def compare_algorithms(
        self,
        algorithms: Dict[str, Callable],
        episodes: int = 20,
        render: bool = False
    ) -> Dict[str, Any]:
        """Compare performance of different algorithms
        
        Args:
            algorithms: Algorithm dictionary, format: {algorithm_name: policy_function}
            episodes: Number of test episodes per algorithm
            render: Whether to render
            
        Returns:
            Comparison results, including performance metrics of different algorithms
        """
        print(f"=== Algorithm Comparison: {self.env_name} ===")
        
        results = {}
        
        for algo_name, policy in algorithms.items():
            print(f"\nTesting algorithm: {algo_name}")
            
            # Run tests
            algo_results = {
                "returns": [],
                "episode_lengths": [],
                "success_rates": [],
                "all_rewards": [],
                "all_actions": []
            }
            
            successes = 0
            
            for ep in range(episodes):
                env = gym.make(self.env_name)
                obs, info = env.reset()
                
                ep_return = 0.0
                ep_steps = 0
                
                ep_rewards = []
                ep_actions = []
                
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
                    
                    ep_rewards.append(reward)
                    ep_actions.append(action)
                    
                    obs = next_obs
                    
                    if render:
                        env.mj_render()
                    
                    if done:
                        # Check if successful
                        if "success" in info and info["success"]:
                            successes += 1
                        break
                
                # Save results
                algo_results["returns"].append(ep_return)
                algo_results["episode_lengths"].append(ep_steps)
                algo_results["all_rewards"].append(ep_rewards)
                algo_results["all_actions"].append(ep_actions)
                
                env.close()
            
            # Calculate statistics
            success_rate = successes / episodes
            avg_return = np.mean(algo_results["returns"])
            std_return = np.std(algo_results["returns"])
            avg_length = np.mean(algo_results["episode_lengths"])
            std_length = np.std(algo_results["episode_lengths"])
            
            algo_results["success_rates"] = success_rate
            
            print(f"  Avg Return: {avg_return:.3f} ± {std_return:.3f}")
            print(f"  Avg Length: {avg_length:.3f} ± {std_length:.3f}")
            print(f"  Success Rate: {success_rate:.3f}")
            
            results[algo_name] = algo_results
        
        # Plot comparison results
        self._plot_algorithm_comparison(results)
        
        return results
    
    def _plot_algorithm_comparison(self, results: Dict[str, Any]):
        """Plot algorithm comparison results"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        algorithms = list(results.keys())
        
        plt.figure(figsize=(15, 10))
        
        # Plot return boxplot
        plt.subplot(2, 2, 1)
        returns = [results[algo]["returns"] for algo in algorithms]
        plt.boxplot(returns, labels=algorithms)
        plt.title("Return Distribution by Algorithm")
        plt.xlabel("Algorithm")
        plt.ylabel("Return")
        plt.grid(True)
        
        # Plot success rate bar chart
        plt.subplot(2, 2, 2)
        success_rates = [results[algo]["success_rates"] for algo in algorithms]
        plt.bar(algorithms, success_rates)
        plt.title("Success Rates by Algorithm")
        plt.xlabel("Algorithm")
        plt.ylabel("Success Rate")
        plt.grid(True)
        
        # Plot average return line chart
        plt.subplot(2, 2, 3)
        for algo in algorithms:
            plt.plot(results[algo]["returns"], label=algo)
        plt.title("Returns per Episode by Algorithm")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.legend()
        plt.grid(True)
        
        # Plot statistical summary
        plt.subplot(2, 2, 4)
        avg_returns = [np.mean(results[algo]["returns"]) for algo in algorithms]
        avg_lengths = [np.mean(results[algo]["episode_lengths"]) for algo in algorithms]
        
        width = 0.35
        x = np.arange(len(algorithms))
        
        plt.bar(x - width/2, avg_returns, width, label='Avg Return')
        plt.bar(x + width/2, avg_lengths, width, label='Avg Episode Length')
        plt.title("Algorithm Performance Summary")
        plt.xlabel("Algorithm")
        plt.ylabel("Value")
        plt.xticks(x, algorithms)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join("figures", f"algorithm_comparison_{self.env_name}.png"))
        plt.close()
        print(f"Algorithm comparison results saved as figures/algorithm_comparison_{self.env_name}.png")
    
    def load_ppo_policy(self, model_path: str) -> Callable:
        """Load PPO policy"""
        from ppo_agent import PPOAgent
        
        env = gym.make(self.env_name)
        agent = PPOAgent(env)
        agent.load(model_path)
        
        def ppo_policy(obs):
            return agent.get_action(obs)
        
        return ppo_policy
    
    def create_simple_heuristic_policy(self) -> Callable:
        """Create simple heuristic policy"""
        env = gym.make(self.env_name)
        act_space = env.action_space
        
        def heuristic_policy(obs):
            # Simple proportional controller: give action along error direction, ensure action dimension matches
            error = obs[:min(3, len(obs))]  # Assume first few dimensions are error
            # Expand or repeat error to match action space dimension
            action = -0.1 * np.tile(error, (act_space.shape[0] // len(error)) + 1)[:act_space.shape[0]]
            # Clip to action space
            return np.clip(action, act_space.low, act_space.high)
        
        return heuristic_policy
    
    def create_zero_policy(self) -> Callable:
        """Create zero policy"""
        env = gym.make(self.env_name)
        act_space = env.action_space
        
        def zero_policy(obs):
            return np.zeros(act_space.shape)
        
        return zero_policy
    
    def create_random_policy(self) -> Callable:
        """Create random policy"""
        env = gym.make(self.env_name)
        act_space = env.action_space
        
        def random_policy(obs):
            return act_space.sample()
        
        return random_policy
    
    def close(self):
        """Close environment"""
        self.base_env.close()
