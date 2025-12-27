#!/usr/bin/env python3
"""
Train SAC model for algorithm comparison
"""

import os
from myosuite.utils import gym
from sac_agent import SACAgent
from utils.env_utils import set_seed


def main():
    """Main function to train SAC model"""
    print("=== Training SAC Model ===")
    
    # Environment name
    env_name = "myoElbowPose1D6MRandom-v0"
    
    # Create environment
    env = gym.make(env_name)
    
    # Set seed for reproducibility
    seed = 0
    set_seed(seed, env)
    
    # Create SAC agent
    agent = SACAgent(env, seed=seed)
    
    # Train the agent
    total_timesteps = 100000  # 100k steps
    print(f"Training SAC agent for {total_timesteps} timesteps...")
    
    stats = agent.run_training(
        total_timesteps=total_timesteps,
        steps_per_update=100,
        updates_per_step=1,
        log_interval=10
    )
    
    # Save the trained model
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, f"sac_model_{total_timesteps}_steps.zip_seed{seed}.pt")
    agent.save(model_path)
    
    print(f"\n=== SAC Model Training Complete ===")
    print(f"Model saved to: {model_path}")
    print(f"Final Average Return: {np.mean(stats['returns'][-10:]):.2f}")
    print(f"Final Average Episode Length: {np.mean(stats['episode_lengths'][-10:]):.1f}")
    
    # Close the environment
    env.close()


if __name__ == "__main__":
    import numpy as np
    main()