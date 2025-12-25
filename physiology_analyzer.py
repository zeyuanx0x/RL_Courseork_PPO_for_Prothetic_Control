import numpy as np
import os
from typing import Dict, List, Callable, Any
from myosuite.utils import gym

# Try to import matplotlib, if it fails, we'll skip plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available, plotting will be skipped")
    MATPLOTLIB_AVAILABLE = False


class PhysiologyAnalyzer:
    """Physiology Rationality and Safety Analyzer"""
    
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.base_env = gym.make(env_name)
        
        # Check if environment has muscle-related information
        self.has_muscle_info = self._check_muscle_info()
    
    def _check_muscle_info(self) -> bool:
        """Check if environment provides muscle-related information"""
        # Try to get muscle information
        try:
            obs, info = self.base_env.reset()
            if "muscle_activation" in info or hasattr(self.base_env, "muscle_activations"):
                return True
        except Exception:
            pass
        return False
    
    def muscle_activation_analysis(
        self,
        policy: Callable,
        episodes: int = 10,
        render: bool = False
    ) -> Dict[str, Any]:
        """Muscle Activation Analysis
        
        Args:
            policy: Policy function to test
            episodes: Number of test episodes
            render: Whether to render
            
        Returns:
            Muscle activation analysis results, including RMS, peaks, synergy patterns, etc.
        """
        print(f"=== Muscle Activation Analysis: {self.env_name} ===")
        
        results = {
            "muscle_activations": [],  # Muscle activation data for all episodes
            "rms_values": [],  # RMS values for each episode
            "peak_values": [],  # Peak values for each episode
            "mean_activations": [],  # Mean values for each episode
        }
        
        env = gym.make(self.env_name)
        
        for ep in range(episodes):
            obs, info = env.reset()
            
            ep_activations = []
            
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
                
                # Get muscle activation data
                muscle_act = self._get_muscle_activation(info, env)
                if muscle_act is not None:
                    ep_activations.append(muscle_act)
                
                obs = next_obs
                
                if render:
                    env.mj_render()
                
                if done:
                    break
            
            if ep_activations:
                ep_activations = np.array(ep_activations)
                results["muscle_activations"].append(ep_activations)
                
                # Calculate RMS values
                rms = np.sqrt(np.mean(ep_activations**2, axis=0))
                results["rms_values"].append(rms)
                
                # Calculate peak values
                peaks = np.max(ep_activations, axis=0)
                results["peak_values"].append(peaks)
                
                # Calculate mean values
                means = np.mean(ep_activations, axis=0)
                results["mean_activations"].append(means)
                
                print(f"  Episode {ep+1}: RMS={rms.mean():.3f}, Peak={peaks.mean():.3f}, Mean={means.mean():.3f}")
        
        env.close()
        
        if results["muscle_activations"]:
            # Calculate statistics for all episodes
            all_rms = np.array(results["rms_values"])
            all_peaks = np.array(results["peak_values"])
            all_means = np.array(results["mean_activations"])
            
            results["overall_stats"] = {
                "avg_rms": np.mean(all_rms, axis=0),
                "avg_peaks": np.mean(all_peaks, axis=0),
                "avg_means": np.mean(all_means, axis=0),
                "std_rms": np.std(all_rms, axis=0),
                "std_peaks": np.std(all_peaks, axis=0),
                "std_means": np.std(all_means, axis=0),
            }
            
            # Plot muscle activation results
            self._plot_muscle_activation(results)
        
        return results
    
    def _get_muscle_activation(self, info: Dict, env: gym.Env) -> np.ndarray:
        """Get muscle activation data from environment or info"""
        # Try to get from info
        if "muscle_activation" in info:
            return info["muscle_activation"]
        elif "muscle_activations" in info:
            return info["muscle_activations"]
        
        # Try to get from environment attributes
        if hasattr(env, "muscle_activations"):
            return env.muscle_activations
        elif hasattr(env, "sim") and hasattr(env.sim, "data") and hasattr(env.sim.data, "act"):
            return env.sim.data.act
        
        # Default: return action as muscle activation (simplified processing)
        return None
    
    def metabolic_cost_analysis(
        self,
        policy: Callable,
        episodes: int = 10
    ) -> Dict[str, List]:
        """Metabolic Cost Analysis
        
        Args:
            policy: Policy function to test
            episodes: Number of test episodes
            
        Returns:
            Metabolic cost analysis results, including muscle activation squared integral, mechanical power, etc.
        """
        print(f"=== Metabolic Cost Analysis: {self.env_name} ===")
        
        results = {
            "activation_cost": [],  # Muscle activation squared integral
            "mechanical_power": [],  # Mechanical power
            "total_energy": [],  # Total energy consumption
        }
        
        env = gym.make(self.env_name)
        
        for ep in range(episodes):
            obs, info = env.reset()
            
            ep_activations = []
            ep_rewards = []
            
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
                
                # Get muscle activation data
                muscle_act = self._get_muscle_activation(info, env)
                if muscle_act is not None:
                    ep_activations.append(muscle_act)
                
                ep_rewards.append(reward)
                
                obs = next_obs
                
                if done:
                    break
            
            if ep_activations:
                # Calculate muscle activation squared integral (common metabolic proxy indicator)
                act_cost = np.sum(np.square(ep_activations))
                
                # Calculate mechanical power (simplified: total reward as proxy)
                mechanical_power = np.sum(ep_rewards)
                
                # Calculate total energy consumption (simplified processing)
                total_energy = act_cost + abs(mechanical_power)
                
                results["activation_cost"].append(act_cost)
                results["mechanical_power"].append(mechanical_power)
                results["total_energy"].append(total_energy)
                
                print(f"  Episode {ep+1}: Activation Cost={act_cost:.3f}, Mechanical Power={mechanical_power:.3f}, Total Energy={total_energy:.3f}")
        
        env.close()
        
        # Plot metabolic cost results
        self._plot_metabolic_cost(results)
        
        return results
    
    def safety_analysis(
        self,
        policy: Callable,
        episodes: int = 10
    ) -> Dict[str, Any]:
        """Safety Analysis
        
        Args:
            policy: Policy function to test
            episodes: Number of test episodes
            
        Returns:
            Safety analysis results, including joint angle violations, contact impacts, etc.
        """
        print(f"=== Safety Analysis: {self.env_name} ===")
        
        results = {
            "joint_angle_violations": [],  # Number of joint angle violations
            "joint_velocity_violations": [],  # Number of joint velocity violations
            "contact_impulse_peaks": [],  # Contact impulse peaks
            "jerk_values": [],  # Jerk values (smoothness indicator)
        }
        
        env = gym.make(self.env_name)
        
        # Get joint limits (if available)
        joint_limits = self._get_joint_limits(env)
        
        for ep in range(episodes):
            obs, info = env.reset()
            
            ep_joint_angles = []
            ep_joint_velocities = []
            ep_impulses = []
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
                
                # Get joint information
                joint_info = self._get_joint_info(info, env)
                if joint_info:
                    ep_joint_angles.append(joint_info["angles"])
                    ep_joint_velocities.append(joint_info["velocities"])
                
                # Get contact impact information
                if "contact_impulse" in info:
                    ep_impulses.append(info["contact_impulse"])
                
                ep_actions.append(action)
                
                obs = next_obs
                
                if done:
                    break
            
            # Calculate safety indicators
            angle_violations = 0
            velocity_violations = 0
            
            if ep_joint_angles and joint_limits:
                ep_joint_angles = np.array(ep_joint_angles)
                ep_joint_velocities = np.array(ep_joint_velocities)
                
                # Check joint angle violations
                for i in range(ep_joint_angles.shape[1]):
                    if i < len(joint_limits["angle_limits"]):
                        min_angle, max_angle = joint_limits["angle_limits"][i]
                        angle_violations += np.sum((ep_joint_angles[:, i] < min_angle) | (ep_joint_angles[:, i] > max_angle))
                
                # Check joint velocity violations
                for i in range(ep_joint_velocities.shape[1]):
                    if i < len(joint_limits["velocity_limits"]):
                        max_vel = joint_limits["velocity_limits"][i]
                        velocity_violations += np.sum(np.abs(ep_joint_velocities[:, i]) > max_vel)
            
            # Calculate contact impulse peaks
            if ep_impulses:
                impulse_peaks = np.max(np.abs(ep_impulses))
            else:
                impulse_peaks = 0.0
            
            # Calculate jerk values (action change rate)
            if len(ep_actions) > 1:
                ep_actions = np.array(ep_actions)
                jerk = np.sum(np.abs(np.diff(ep_actions, n=2, axis=0)))
            else:
                jerk = 0.0
            
            results["joint_angle_violations"].append(angle_violations)
            results["joint_velocity_violations"].append(velocity_violations)
            results["contact_impulse_peaks"].append(impulse_peaks)
            results["jerk_values"].append(jerk)
            
            print(f"  Episode {ep+1}: Angle Violations={angle_violations}, Velocity Violations={velocity_violations}, "
                  f"Impulse Peaks={impulse_peaks:.3f}, Jerk={jerk:.3f}")
        
        env.close()
        
        # Plot safety analysis results
        self._plot_safety_analysis(results)
        
        return results
    
    def _get_joint_info(self, info: Dict, env: gym.Env) -> Dict[str, np.ndarray]:
        """Get joint information"""
        # Simplified processing: assume observation contains joint angles and velocities
        # Need to adjust based on specific environment's observation space structure
        if len(info) > 0:
            # Try to get from info
            if "joint_angles" in info and "joint_velocities" in info:
                return {
                    "angles": info["joint_angles"],
                    "velocities": info["joint_velocities"]
                }
        
        # Default: return None
        return None
    
    def _get_joint_limits(self, env: gym.Env) -> Dict[str, List]:
        """Get joint limits"""
        # Simplified processing: return default limits
        return {
            "angle_limits": [(-np.pi, np.pi)] * 6,  # Assume 6 joints
            "velocity_limits": [10.0] * 6
        }
    
    def _plot_muscle_activation(self, results: Dict[str, Any]):
        """Plot muscle activation results"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        plt.figure(figsize=(15, 10))
        
        # Plot average RMS values
        plt.subplot(3, 1, 1)
        avg_rms = np.mean(results["rms_values"], axis=0)
        std_rms = np.std(results["rms_values"], axis=0)
        plt.errorbar(range(len(avg_rms)), avg_rms, yerr=std_rms, fmt='o-')
        plt.title("Average Muscle Activation RMS")
        plt.xlabel("Muscle Index")
        plt.ylabel("RMS Activation")
        plt.grid(True)
        
        # Plot average peak values
        plt.subplot(3, 1, 2)
        avg_peaks = np.mean(results["peak_values"], axis=0)
        std_peaks = np.std(results["peak_values"], axis=0)
        plt.errorbar(range(len(avg_peaks)), avg_peaks, yerr=std_peaks, fmt='o-')
        plt.title("Average Muscle Activation Peaks")
        plt.xlabel("Muscle Index")
        plt.ylabel("Peak Activation")
        plt.grid(True)
        
        # Plot average activation
        plt.subplot(3, 1, 3)
        avg_means = np.mean(results["mean_activations"], axis=0)
        std_means = np.std(results["mean_activations"], axis=0)
        plt.errorbar(range(len(avg_means)), avg_means, yerr=std_means, fmt='o-')
        plt.title("Average Muscle Activation")
        plt.xlabel("Muscle Index")
        plt.ylabel("Mean Activation")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join("figures", f"muscle_activation_{self.env_name}.png"))
        plt.close()
        print(f"Muscle activation results saved as figures/muscle_activation_{self.env_name}.png")
    
    def _plot_metabolic_cost(self, results: Dict[str, List]):
        """Plot metabolic cost results"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        plt.figure(figsize=(15, 5))
        
        # Plot metabolic cost indicators
        plt.subplot(1, 3, 1)
        plt.plot(results["activation_cost"])
        plt.title("Muscle Activation Cost")
        plt.xlabel("Episode")
        plt.ylabel("Activation Cost")
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(results["mechanical_power"])
        plt.title("Mechanical Power")
        plt.xlabel("Episode")
        plt.ylabel("Mechanical Power")
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(results["total_energy"])
        plt.title("Total Energy Consumption")
        plt.xlabel("Episode")
        plt.ylabel("Total Energy")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join("figures", f"metabolic_cost_{self.env_name}.png"))
        plt.close()
        print(f"Metabolic cost results saved as figures/metabolic_cost_{self.env_name}.png")
    
    def _plot_safety_analysis(self, results: Dict[str, List]):
        """Plot safety analysis results"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        plt.figure(figsize=(15, 10))
        
        # Plot joint angle violations
        plt.subplot(2, 2, 1)
        plt.plot(results["joint_angle_violations"])
        plt.title("Joint Angle Violations")
        plt.xlabel("Episode")
        plt.ylabel("Number of Violations")
        plt.grid(True)
        
        # Plot joint velocity violations
        plt.subplot(2, 2, 2)
        plt.plot(results["joint_velocity_violations"])
        plt.title("Joint Velocity Violations")
        plt.xlabel("Episode")
        plt.ylabel("Number of Violations")
        plt.grid(True)
        
        # Plot contact impulse peaks
        plt.subplot(2, 2, 3)
        plt.plot(results["contact_impulse_peaks"])
        plt.title("Contact Impulse Peaks")
        plt.xlabel("Episode")
        plt.ylabel("Impulse Peak")
        plt.grid(True)
        
        # Plot jerk values
        plt.subplot(2, 2, 4)
        plt.plot(results["jerk_values"])
        plt.title("Action Jerk")
        plt.xlabel("Episode")
        plt.ylabel("Jerk")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join("figures", f"safety_analysis_{self.env_name}.png"))
        plt.close()
        print(f"Safety analysis results saved as figures/safety_analysis_{self.env_name}.png")
    
    def close(self):
        """Close the environment"""
        self.base_env.close()
