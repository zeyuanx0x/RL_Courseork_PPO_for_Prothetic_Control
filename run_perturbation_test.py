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

# 4. Create Perturbation tester
tester = RobustnessTester(env_name)

# 5. Define perturbation types and strengths
perturbation_types = ["force", "noise", "delay"]
perturbation_strengths = [0.1, 0.2, 0.3]

# 6. Execute perturbation experiments
all_results = []

try:
    for p_type in perturbation_types:
        for strength in perturbation_strengths:
            print(f"\n=== Running {p_type} perturbation with strength {strength} ===")
            
            # Execute perturbation test
            results = tester.perturbation_test(
                policy=ppo_policy,
                perturbation_type=p_type,
                perturbation_strength=strength,
                episodes=10
            )
            
            # Save results
            for ep_idx in range(len(results["returns"])):
                all_results.append({
                    "algo": "PPO",
                    "perturb_type": p_type,
                    "perturb_strength": strength,
                    "episode_idx": ep_idx,
                    "episode_return": results["returns"][ep_idx],
                    "episode_len": results["episode_lengths"][ep_idx],
                    "recovery_time": results["recovery_times"][ep_idx],
                    "max_deviation": results["max_deviations"][ep_idx]
                })
    
    # 7. Save all results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/perturbation_results.csv", index=False)
    print("\n=== Perturbation Experiment Complete ===")
    print("Results saved to: results/perturbation_results.csv")
    
except Exception as e:
    print(f"Error running perturbation test: {e}")
    print("Perturbation experiment failed due to missing perturbation logging or runner issues.")
    
finally:
    # Close environment
    tester.close()
env.close()
