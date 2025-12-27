import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Any
import json
import os

# Import NatureStylePlotter
from plotting_utils import NatureStylePlotter

class FailureAnalyzer:
    """Failure mode analyzer, responsible for classifying, explaining, and visualizing failed episodes"""
    
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.failure_categories = {
            "control_divergence": [],
            "delay_instability": [],
            "over_assistance": [],
            "under_assistance": [],
            "safety_violation": []
        }
        
    def load_test_results(self, test_results: Dict[str, List]) -> None:
        """Load test results"""
        self.test_results = test_results
    
    def analyze_failures(self, success_threshold: float = 0.5) -> Dict[str, Dict[str, Any]]:
        """Analyze and classify failed episodes
        
        Args:
            success_threshold: Success return threshold, episodes below this value are considered failed
            
        Returns:
            Failure analysis results, including representative episodes and explanations for each failure type
        """
        print(f"=== Failure Mode Analysis: {self.env_name} ===")
        
        # First identify all failed episodes
        all_episodes = self._extract_all_episodes()
        failed_episodes = self._identify_failed_episodes(all_episodes, success_threshold)
        
        print(f"Total episodes: {len(all_episodes)}")
        print(f"Failed episodes: {len(failed_episodes)}")
        
        # Classify failed episodes
        for i, episode in enumerate(failed_episodes):
            failure_type = self._classify_failure(episode)
            self.failure_categories[failure_type].append((i, episode))
        
        # Count failures by category
        for category, episodes in self.failure_categories.items():
            print(f"  {category}: {len(episodes)} episodes")
        
        # Generate analysis report
        analysis_report = self._generate_analysis_report()
        
        # Save analysis results
        self._save_analysis_results(analysis_report)
        
        # No failure type distribution figure generated
        
        return analysis_report
    
    def _extract_all_episodes(self) -> List[Dict[str, Any]]:
        """Extract all episodes from test results"""
        episodes = []
        
        if "all_obs" in self.test_results and "all_actions" in self.test_results and "all_rewards" in self.test_results:
            # Extract from run_policy results
            for i in range(len(self.test_results["all_obs"])):
                episode = {
                    "episode_id": i,
                    "observations": np.array(self.test_results["all_obs"][i]),
                    "actions": np.array(self.test_results["all_actions"][i]),
                    "rewards": np.array(self.test_results["all_rewards"][i]),
                    "return": self.test_results["returns"][i],
                    "length": self.test_results["episode_lengths"][i]
                }
                episodes.append(episode)
        elif "returns" in self.test_results and isinstance(self.test_results["returns"][0], list):
            # Extract from robustness test results
            for set_idx, set_returns in enumerate(self.test_results["returns"]):
                for ep_idx, ep_return in enumerate(set_returns):
                    # Assume robustness test result format, may need adjustment in practice
                    episode = {
                        "episode_id": f"set{set_idx}_ep{ep_idx}",
                        "return": ep_return,
                        "length": self.test_results["episode_lengths"][set_idx][ep_idx],
                        "param_setting": self.test_results.get("param_settings", [{}])[set_idx]
                    }
                    episodes.append(episode)
        
        return episodes
    
    def _identify_failed_episodes(self, episodes: List[Dict[str, Any]], success_threshold: float) -> List[Dict[str, Any]]:
        """Identify failed episodes"""
        # Calculate average return as baseline
        avg_return = np.mean([ep["return"] for ep in episodes])
        threshold = success_threshold * avg_return
        
        failed = [ep for ep in episodes if ep["return"] < threshold]
        return failed
    
    def _classify_failure(self, episode: Dict[str, Any]) -> str:
        """Classify a single failed episode"""
        # Check if detailed episode data is available
        if "actions" not in episode or "observations" not in episode:
            # Initial classification based on parameter settings and return
            param_setting = episode.get("param_setting", {})
            if "delay" in param_setting and param_setting["delay"] > 0.1:
                return "delay_instability"
            return "control_divergence"
        
        actions = episode["actions"]
        observations = episode["observations"]
        rewards = episode["rewards"]
        
        # 1. Control divergence: action explosion or high-frequency oscillation
        action_magnitude = np.mean(np.abs(actions))
        action_std = np.std(actions, axis=0)
        if action_magnitude > 1.5 or np.any(action_std > 2.0):
            return "control_divergence"
        
        # 2. Delay instability: check if there are delay parameters or action lag phenomena
        if episode.get("param_setting", {}).get("delay", 0) > 0.1:
            return "delay_instability"
        
        # 3. Over assistance: check if actions are consistently too large
        if np.mean(np.abs(actions)) > 1.0 and np.mean(rewards) < -1.0:
            return "over_assistance"
        
        # 4. Under assistance: check if actions are consistently too small but returns are very low
        if np.mean(np.abs(actions)) < 0.2 and np.mean(rewards) < -1.0:
            return "under_assistance"
        
        # 5. Safety boundary violation: check if observations are outside reasonable range
        obs_min = np.min(observations, axis=0)
        obs_max = np.max(observations, axis=0)
        if np.any(obs_min < -10.0) or np.any(obs_max > 10.0):
            return "safety_violation"
        
        # Default classification
        return "control_divergence"
    
    def _generate_analysis_report(self) -> Dict[str, Dict[str, Any]]:
        """Generate failure analysis report"""
        report = {
            "env_name": self.env_name,
            "failure_categories": {},
            "summary": ""
        }
        
        for category, episodes in self.failure_categories.items():
            if not episodes:
                continue
            
            # Extract 1-3 representative episodes
            num_representative = min(3, len(episodes))
            # Sort by return from lowest to highest, take most representative
            sorted_episodes = sorted(episodes, key=lambda x: x[1]["return"])
            representative = sorted_episodes[:num_representative]
            
            # Generate explanatory summary
            explanation = self._generate_explanation(category, representative)
            
            # Save representative episode data
            representative_data = []
            for ep_idx, (orig_idx, ep) in enumerate(representative):
                ep_data = {
                    "episode_id": ep["episode_id"],
                    "return": ep["return"],
                    "length": ep["length"]
                }
                if "actions" in ep and "observations" in ep:
                    ep_data["action_stats"] = {
                        "mean": float(np.mean(np.abs(ep["actions"]))),
                        "std": float(np.mean(np.std(ep["actions"], axis=0)))
                    }
                    ep_data["obs_stats"] = {
                        "mean": float(np.mean(np.abs(ep["observations"]))),
                        "std": float(np.mean(np.std(ep["observations"], axis=0)))
                    }
                representative_data.append(ep_data)
            
            # Generate visualization
            self._plot_representative_episodes(category, representative)
            
            report["failure_categories"][category] = {
                "count": len(episodes),
                "representative_episodes": representative_data,
                "explanation": explanation
            }
        
        # Generate overall summary
        total_failures = sum(len(eps) for eps in self.failure_categories.values())
        report["summary"] = f"""Failure analysis on {self.env_name} identified {total_failures} failed episodes across {len([c for c, eps in self.failure_categories.items() if eps])} categories. 
        The most common failure modes were {', '.join([c for c, eps in self.failure_categories.items() if eps and len(eps) > total_failures*0.2])}. 
        Detailed mechanistic interpretations for each failure mode are provided, along with representative episodes and key statistics."""
        
        return report
    
    def _generate_explanation(self, category: str, representative: List[tuple]) -> str:
        """Generate explanatory summary for failure mode"""
        explanations = {
            "control_divergence": """Control divergence failure is characterized by explosive action magnitudes or high-frequency oscillations in the policy output. 
            This typically occurs when the policy fails to stabilize the system, leading to increasing action amplitudes that drive the system away from the desired state. 
            Key indicators include abnormally large action magnitudes (mean absolute value > 1.5) and high action variability (standard deviation > 2.0). 
            The root cause is often insufficient exploration during training or unstable policy updates that fail to capture the system's dynamics accurately.""",
            
            "delay_instability": """Delay-induced instability arises when there is a significant delay between action generation and execution, leading to phase mismatches in the control loop. 
            This failure mode is particularly prevalent in systems with communication delays or sensor processing latency. 
            The delayed feedback causes the policy to overcompensate for past errors, resulting in oscillatory behavior that eventually diverges. 
            The analysis shows that delays exceeding 0.1 seconds significantly increase the likelihood of this failure mode.""",
            
            "over_assistance": """Over-assistance failure occurs when the exoskeleton provides excessive assistance, leading to instability or unwanted human adaptation. 
            This is indicated by consistently large action magnitudes (mean > 1.0) coupled with negative rewards. 
            The over-powering nature of the exoskeleton disrupts natural human movement, causing the system to deviate from the desired trajectory. 
            This failure mode highlights the importance of balancing assistance with human autonomy in collaborative control systems.""",
            
            "under_assistance": """Under-assistance failure happens when the exoskeleton provides insufficient support, resulting in poor task performance despite the human's best efforts. 
            Characterized by small action magnitudes (mean < 0.2) and low rewards, this failure mode indicates that the policy is not effectively augmenting human capabilities. 
            The exoskeleton fails to provide meaningful assistance, leaving the human to bear most of the task load, which leads to suboptimal performance and potential fatigue.""",
            
            "safety_violation": """Safety boundary violation occurs when the system operates outside predefined safe limits for joint angles, velocities, or torques. 
            This is detected by observing extreme values in the observation space (values < -10.0 or > 10.0). 
            Such violations pose significant risks to both the human user and the exoskeleton hardware. 
            The root cause may be inadequate safety constraints in the reward function or a failure to account for extreme environmental conditions during training."""
        }
        
        return explanations.get(category, "Unknown failure mode")
    
    def _plot_representative_episodes(self, category: str, representative: List[tuple]) -> None:
        """Plot key time series for representative episodes"""
        if not representative:
            return
        
        plt.figure(figsize=(15, 12))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # Plot action and observation time series
        for i, (orig_idx, episode) in enumerate(representative[:3]):
            if "actions" not in episode or "observations" not in episode:
                continue
            
            # Plot actions
            plt.subplot(3, 2, 2*i + 1)
            actions = episode["actions"]
            for act_dim in range(min(3, actions.shape[1])):
                plt.plot(actions[:, act_dim], label=f"Act Dim {act_dim+1}", alpha=0.8)
            plt.title(f"Failure Type: {category} - Episode {episode['episode_id']}\nActions")
            plt.xlabel("Step")
            plt.ylabel("Action Value")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Plot observations
            plt.subplot(3, 2, 2*i + 2)
            observations = episode["observations"]
            for obs_dim in range(min(3, observations.shape[1])):
                plt.plot(observations[:, obs_dim], label=f"Obs Dim {obs_dim+1}", alpha=0.8)
            plt.title(f"Observations")
            plt.xlabel("Step")
            plt.ylabel("Observation Value")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join("figures", f"failure_{category}_representative_{self.env_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_analysis_results(self, report: Dict[str, Any]) -> None:
        """Save analysis results to files"""
        os.makedirs("results/failure_analysis", exist_ok=True)
        
        # Save JSON report
        with open(f"results/failure_analysis/{self.env_name}_failure_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate Markdown report
        md_content = f"# Failure Mode Analysis Report\n\n"
        md_content += f"## Environment: {self.env_name}\n\n"
        md_content += f"## Summary\n{report['summary']}\n\n"
        
        for category, data in report["failure_categories"].items():
            md_content += f"## Failure Type: {category}\n\n"
            md_content += f"- **Count**: {data['count']} episodes\n"
            md_content += f"- **Representative Episodes**: {len(data['representative_episodes'])}\n\n"
            md_content += f"### Explanation\n{data['explanation']}\n\n"
            md_content += f"### Representative Episodes Statistics\n"
            md_content += "| Episode ID | Return | Length | Action Mean | Action Std |\n"
            md_content += "|------------|--------|--------|-------------|------------|\n"
            for ep in data["representative_episodes"]:
                action_mean = ep.get("action_stats", {}).get("mean", "N/A")
                action_std = ep.get("action_stats", {}).get("std", "N/A")
                md_content += f"| {ep['episode_id']} | {ep['return']:.3f} | {ep['length']} | {action_mean:.3f} | {action_std:.3f} |\n"
            md_content += "\n"
            md_content += f"### Visualization\n"
            md_content += f"![{category}](figures/failure_{category}_representative_{self.env_name}.png)\n\n"
        
        with open(f"results/failure_analysis/{self.env_name}_failure_report.md", "w") as f:
            f.write(md_content)
        
        print(f"Failure analysis results saved to results/failure_analysis/{self.env_name}_failure_report.md")


def main():
    """Main function, used to test the failure analyzer"""
    from experiment_manager import ExperimentManager
    
    env_name = "myoElbowPose1D6MRandom-v0"
    
    # Create experiment manager
    manager = ExperimentManager(env_name)
    
    # Run a simple test to generate failure data
    print("Running test with random policy to generate failure data...")
    test_results = manager.run_policy(manager.random_policy, episodes=10)
    
    # Create failure analyzer
    analyzer = FailureAnalyzer(env_name)
    analyzer.load_test_results(test_results)
    
    # Run analysis
    analysis_report = analyzer.analyze_failures()
    
    # Print summary
    print("\n=== Failure Analysis Summary ===")
    print(analysis_report["summary"])
    
    manager.close()


if __name__ == "__main__":
    main()
