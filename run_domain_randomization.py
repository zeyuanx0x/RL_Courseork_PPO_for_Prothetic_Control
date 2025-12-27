import numpy as np
import pandas as pd
import os
from robustness_tester import RobustnessTester
from ppo_agent import PPOAgent
from myosuite.utils import gym

# Ensure results directories exist
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

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

# 3. Define policy function
def ppo_policy(obs):
    return agent.get_action(obs)

# 4. Create Domain randomization tester
tester = RobustnessTester(env_name)

# 5. Define randomization parameter ranges
randomization_params = {
    "delay": [0.0, 0.3],  # Delay parameter
    "noise": [0.0, 0.2],  # Noise parameter
    "force": [0.0, 0.1]   # External force parameter
}

# 6. Execute Domain randomization test
print("\n=== Running Domain Randomization Test ===")
results = tester.domain_randomization_test(
    policy=ppo_policy,
    randomization_params=randomization_params,
    episodes=10
)

# 7. Save Domain randomization results to CSV
domain_results = []

# Parse test results
for i, (param_setting, ep_returns, ep_lengths, success_rate) in enumerate(zip(
    results["param_settings"],
    results["returns"],
    results["episode_lengths"],
    results["success_rates"]
)):
    # Save results for each episode
    for ep_idx in range(len(ep_returns)):
        result = {
            "param_setting_id": i,
            "episode_idx": ep_idx,
            "episode_return": ep_returns[ep_idx],
            "episode_len": ep_lengths[ep_idx],
            "success_rate": success_rate
        }
        
        # Add randomization parameter values
        result.update(param_setting)
        
        domain_results.append(result)

# Save to CSV
results_df = pd.DataFrame(domain_results)
results_df.to_csv("results/domain_randomization_results.csv", index=False)

print("\n=== Domain Randomization Test Complete ===")
print("Results saved to: results/domain_randomization_results.csv")

# 8. Close environment
tester.close()
env.close()
