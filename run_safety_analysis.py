import numpy as np
import pandas as pd
import os
from failure_analysis import FailureAnalyzer
from experiment_manager import ExperimentManager
from ppo_agent import PPOAgent
from myosuite.utils import gym

# Ensure results directories exist
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("results/failure_analysis", exist_ok=True)

# 1. Initialize environment and Agent
env_name = "myoElbowPose1D6MRandom-v0"
env = gym.make(env_name)
agent = PPOAgent(env)

# 2. Load pre-trained model
checkpoint_path = "checkpoints/ppo_myoElbowPose1D6MRandom-v0_seed42.pt"
try:
    agent.load(checkpoint_path)
    print(f"Successfully loaded model: {checkpoint_path}")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

# 3. Create experiment manager
manager = ExperimentManager(env_name)

# 4. Define PPO policy function
def ppo_policy(obs):
    return agent.get_action(obs)

# 5. Run policy to collect test results
print("\n=== Running PPO policy to collect test results ===")
test_results = manager.run_policy(
    policy=ppo_policy,
    episodes=10,
    render=False
)

# 6. Create Safety/Failure analyzer
analyzer = FailureAnalyzer(env_name)

# 7. Load test results
analyzer.load_test_results(test_results)

# 8. Execute failure analysis
analysis_report = analyzer.analyze_failures(success_threshold=0.5)

# 9. Save failure analysis results to CSV
# Extract all episode data
all_episodes = analyzer._extract_all_episodes()
failed_episodes = analyzer._identify_failed_episodes(all_episodes, success_threshold=0.5)

# Create a DataFrame to save safety metrics
safety_data = []
for i, episode in enumerate(all_episodes):
    # Check if it's a failed episode
    is_failed = any(f_ep["episode_id"] == episode["episode_id"] for f_ep in failed_episodes)
    
    # Calculate safety metrics
    safety_metrics = {
        "episode_id": episode["episode_id"],
        "return": episode["return"],
        "is_failed": is_failed,
        "episode_len": episode["length"]
    }
    
    # If detailed episode data exists, calculate more safety metrics
    if "actions" in episode and "observations" in episode:
        actions = episode["actions"]
        observations = episode["observations"]
        
        # Calculate action and observation statistics
        safety_metrics.update({
            "action_mean": float(np.mean(np.abs(actions))),
            "action_std": float(np.mean(np.std(actions, axis=0))),
            "obs_mean": float(np.mean(np.abs(observations))),
            "obs_std": float(np.mean(np.std(observations, axis=0))),
            "max_action": float(np.max(np.abs(actions))),
            "max_obs": float(np.max(np.abs(observations)))
        })
    
    safety_data.append(safety_metrics)

# Save to CSV
safety_df = pd.DataFrame(safety_data)
safety_df.to_csv("results/safety_analysis.csv", index=False)
print("\n=== Safety/Failure Analysis Complete ===")
print("Safety metrics saved to: results/safety_analysis.csv")

# 10. Close environment
manager.close()
env.close()
