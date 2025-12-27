import numpy as np
import os
from typing import Dict, List, Callable, Any, Tuple
from myosuite.utils import gym

# Try to import matplotlib, if it fails, we'll skip plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available, plotting will be skipped")
    MATPLOTLIB_AVAILABLE = False


class ExoskeletonAnalyzer:
    """Exoskeleton Analyzer for human-machine collaboration analysis and assistance ratio scanning"""
    
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.base_env = gym.make(env_name)
        
        # Check if environment is an exoskeleton environment
        self.is_exoskeleton_env = "exo" in env_name.lower() or "Exo" in env_name
    
    def contribution_decomposition(
        self,
        policy: Callable,
        episodes: int = 10,
        render: bool = False
    ) -> Dict[str, Any]:
        """Contribution decomposition: Analyze contributions from human and exoskeleton
        
        Args:
            policy: Policy function to test
            episodes: Number of test episodes
            render: Whether to render
            
        Returns:
            Contribution decomposition results, including performance and force data under different control modes
        """
        print(f"=== Contribution Decomposition: {self.env_name} ===")
        
        # Define three control modes
        control_modes = [
            {"name": "human-only", "human_scale": 1.0, "exo_scale": 0.0},
            {"name": "exo-only", "human_scale": 0.0, "exo_scale": 1.0},
            {"name": "human+exo", "human_scale": 1.0, "exo_scale": 1.0}
        ]
        
        results = {}
        
        for mode in control_modes:
            print(f"\nTesting mode: {mode['name']}")
            
            # Create environment with control mode parameters
            env = gym.make(self.env_name, 
                         human_contribution_scale=mode["human_scale"],
                         exoskeleton_contribution_scale=mode["exo_scale"])
            
            # Run tests
            mode_results = {
                "returns": [],
                "episode_lengths": [],
                "human_forces": [],
                "exo_forces": [],
                "success_rates": []
            }
            
            successes = 0
            
            for ep in range(episodes):
                obs, info = env.reset()
                
                ep_return = 0.0
                ep_steps = 0
                
                ep_human_forces = []
                ep_exo_forces = []
                
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
                    
                    # Get human and exoskeleton force information
                    human_force, exo_force = self._get_forces(info, env)
                    if human_force is not None:
                        ep_human_forces.append(human_force)
                    if exo_force is not None:
                        ep_exo_forces.append(exo_force)
                    
                    obs = next_obs
                    
                    if render:
                        env.mj_render()
                    
                    if done:
                        # Check if successful
                        if "success" in info and info["success"]:
                            successes += 1
                        break
                
                # Save results
                mode_results["returns"].append(ep_return)
                mode_results["episode_lengths"].append(ep_steps)
                if ep_human_forces:
                    mode_results["human_forces"].append(np.mean(ep_human_forces))
                if ep_exo_forces:
                    mode_results["exo_forces"].append(np.mean(ep_exo_forces))
            
            # Calculate success rate
            success_rate = successes / episodes
            mode_results["success_rates"] = success_rate
            
            # Calculate statistical information
            avg_return = np.mean(mode_results["returns"])
            avg_length = np.mean(mode_results["episode_lengths"])
            avg_human_force = np.mean(mode_results["human_forces"]) if mode_results["human_forces"] else 0.0
            avg_exo_force = np.mean(mode_results["exo_forces"]) if mode_results["exo_forces"] else 0.0
            
            print(f"  Avg Return: {avg_return:.3f} | Avg Length: {avg_length:.3f} | Success Rate: {success_rate:.3f}")
            print(f"  Avg Human Force: {avg_human_force:.3f} | Avg Exo Force: {avg_exo_force:.3f}")
            
            results[mode["name"]] = mode_results
            
            env.close()
        
        # Plot contribution decomposition results
        self._plot_contribution_decomposition(results)
        
        return results
    
    def assistance_sweep(
        self,
        policy: Callable,
        assistance_levels: List[float] = None,
        episodes: int = 5,
        render: bool = False
    ) -> Dict[str, Any]:
        """Assistance sweep: Test performance under different assistance levels
        
        Args:
            policy: Policy function to test
            assistance_levels: List of assistance levels, range [0, 1]
            episodes: Number of test episodes per assistance level
            render: Whether to render
            
        Returns:
            Assistance sweep results, including performance under different assistance levels
        """
        print(f"=== Assistance Sweep: {self.env_name} ===")
        
        # Default assistance levels: 0% to 100%, 10% step
        if assistance_levels is None:
            assistance_levels = np.linspace(0.0, 1.0, 11)
        
        results = {
            "assistance_levels": assistance_levels,
            "returns": [],
            "episode_lengths": [],
            "human_forces": [],
            "exo_forces": [],
            "success_rates": []
        }
        
        for assist_level in assistance_levels:
            print(f"\nTesting assistance level: {assist_level:.1f} ({assist_level*100:.0f}%)")
            
            # Create environment with specified assistance level
            env = gym.make(self.env_name, assistance_level=assist_level)
            
            # Run tests
            ep_returns = []
            ep_lengths = []
            ep_human_forces = []
            ep_exo_forces = []
            successes = 0
            
            for ep in range(episodes):
                obs, info = env.reset()
                
                ep_return = 0.0
                ep_steps = 0
                
                human_forces = []
                exo_forces = []
                
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
                    
                    # Get human and exoskeleton force information
                    human_force, exo_force = self._get_forces(info, env)
                    if human_force is not None:
                        human_forces.append(human_force)
                    if exo_force is not None:
                        exo_forces.append(exo_force)
                    
                    obs = next_obs
                    
                    if render:
                        env.mj_render()
                    
                    if done:
                        # Check if successful
                        if "success" in info and info["success"]:
                            successes += 1
                        break
                
                ep_returns.append(ep_return)
                ep_lengths.append(ep_steps)
                if human_forces:
                    ep_human_forces.append(np.mean(human_forces))
                if exo_forces:
                    ep_exo_forces.append(np.mean(exo_forces))
            
            # Calculate statistical information
            avg_return = np.mean(ep_returns)
            avg_length = np.mean(ep_lengths)
            success_rate = successes / episodes
            avg_human_force = np.mean(ep_human_forces) if ep_human_forces else 0.0
            avg_exo_force = np.mean(ep_exo_forces) if ep_exo_forces else 0.0
            
            print(f"  Avg Return: {avg_return:.3f} | Avg Length: {avg_length:.3f} | Success Rate: {success_rate:.3f}")
            print(f"  Avg Human Force: {avg_human_force:.3f} | Avg Exo Force: {avg_exo_force:.3f}")
            
            # Save results
            results["returns"].append(avg_return)
            results["episode_lengths"].append(avg_length)
            results["human_forces"].append(avg_human_force)
            results["exo_forces"].append(avg_exo_force)
            results["success_rates"].append(success_rate)
            
            env.close()
        
        # Plot assistance sweep results
        self._plot_assistance_sweep(results)
        
        return results
    
    def _get_forces(self, info: Dict, env: gym.Env) -> Tuple[float, float]:
        """Get human and exoskeleton force information
        
        Returns:
            (human_force, exo_force) - Forces from human and exoskeleton
        """
        # Simplified processing: Assume info contains this information
        human_force = info.get("human_force", None)
        exo_force = info.get("exo_force", None)
        
        # If not in info, try to get from environment attributes
        if human_force is None and hasattr(env, "human_force"):
            human_force = env.human_force
        if exo_force is None and hasattr(env, "exo_force"):
            exo_force = env.exo_force
        
        return human_force, exo_force
    
    def _plot_contribution_decomposition(self, results: Dict[str, Any]):
        """Plot contribution decomposition results"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        modes = list(results.keys())
        
        plt.figure(figsize=(15, 10))
        
        # Plot returns under different modes
        plt.subplot(2, 2, 1)
        returns = [np.mean(results[mode]["returns"]) for mode in modes]
        plt.bar(modes, returns)
        plt.title("Average Returns by Control Mode")
        plt.xlabel("Control Mode")
        plt.ylabel("Return")
        plt.grid(True)
        
        # Plot success rates under different modes
        plt.subplot(2, 2, 2)
        success_rates = [results[mode]["success_rates"] for mode in modes]
        plt.bar(modes, success_rates)
        plt.title("Success Rates by Control Mode")
        plt.xlabel("Control Mode")
        plt.ylabel("Success Rate")
        plt.grid(True)
        
        # Plot human forces under different modes
        plt.subplot(2, 2, 3)
        human_forces = [np.mean(results[mode]["human_forces"]) if results[mode]["human_forces"] else 0.0 for mode in modes]
        plt.bar(modes, human_forces)
        plt.title("Average Human Force by Control Mode")
        plt.xlabel("Control Mode")
        plt.ylabel("Human Force")
        plt.grid(True)
        
        # Plot exoskeleton forces under different modes
        plt.subplot(2, 2, 4)
        exo_forces = [np.mean(results[mode]["exo_forces"]) if results[mode]["exo_forces"] else 0.0 for mode in modes]
        plt.bar(modes, exo_forces)
        plt.title("Average Exoskeleton Force by Control Mode")
        plt.xlabel("Control Mode")
        plt.ylabel("Exoskeleton Force")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join("figures", f"contribution_decomposition_{self.env_name}.png"))
        plt.close()
        print(f"Contribution decomposition results saved as figures/contribution_decomposition_{self.env_name}.png")
    
    def _plot_assistance_sweep(self, results: Dict[str, Any]):
        """Plot assistance sweep results"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        assistance_levels = results["assistance_levels"]
        
        plt.figure(figsize=(15, 10))
        
        # Plot returns vs assistance level
        plt.subplot(2, 2, 1)
        plt.plot(assistance_levels * 100, results["returns"], marker='o')
        plt.title("Returns vs Assistance Level")
        plt.xlabel("Assistance Level (%)")
        plt.ylabel("Return")
        plt.grid(True)
        
        # Plot success rates vs assistance level
        plt.subplot(2, 2, 2)
        plt.plot(assistance_levels * 100, results["success_rates"], marker='o')
        plt.title("Success Rates vs Assistance Level")
        plt.xlabel("Assistance Level (%)")
        plt.ylabel("Success Rate")
        plt.grid(True)
        
        # Plot human forces vs assistance level
        plt.subplot(2, 2, 3)
        plt.plot(assistance_levels * 100, results["human_forces"], marker='o', label="Human Force")
        plt.plot(assistance_levels * 100, results["exo_forces"], marker='o', label="Exoskeleton Force")
        plt.title("Forces vs Assistance Level")
        plt.xlabel("Assistance Level (%)")
        plt.ylabel("Force")
        plt.legend()
        plt.grid(True)
        
        # Plot episode length vs assistance level
        plt.subplot(2, 2, 4)
        plt.plot(assistance_levels * 100, results["episode_lengths"], marker='o')
        plt.title("Episode Lengths vs Assistance Level")
        plt.xlabel("Assistance Level (%)")
        plt.ylabel("Episode Length")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join("figures", f"assistance_sweep_{self.env_name}.png"))
        plt.close()
        print(f"Assistance sweep results saved as figures/assistance_sweep_{self.env_name}.png")
    
    def close(self):
        """Close the environment"""
        self.base_env.close()
