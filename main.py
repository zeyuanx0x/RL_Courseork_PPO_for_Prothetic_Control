import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List
from myosuite.utils import gym

from experiment_manager import ExperimentManager
from ppo_agent import PPOAgent


def plot_training_curve(stats: Dict[str, List], title: str, env_name: str):
    """Plot training curves"""
    # Create figures directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    
    save_path = os.path.join("figures", f"ppo_training_curve_{env_name}.png")
    
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
    plt.suptitle(f"PPO Training Curves: {env_name}", fontsize=16, fontweight='bold', y=0.98)
    
    # Plot return curve
    plt.subplot(3, 2, 1)
    plt.plot(stats["returns"], color='#1f77b4', linewidth=1.5)
    plt.title("(a) Episode Returns", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=11)
    plt.ylabel("Return", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot episode lengths
    plt.subplot(3, 2, 2)
    plt.plot(stats["episode_lengths"], color='#ff7f0e', linewidth=1.5)
    plt.title("(b) Episode Lengths", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=11)
    plt.ylabel("Length", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot KL divergence
    plt.subplot(3, 2, 3)
    plt.plot(stats["kl_divs"], color='#2ca02c', linewidth=1.5)
    plt.title("(c) KL Divergence", fontsize=12, fontweight='bold')
    plt.xlabel("Update", fontsize=11)
    plt.ylabel("KL Divergence", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot clip fraction
    plt.subplot(3, 2, 4)
    plt.plot(stats["clip_fractions"], color='#d62728', linewidth=1.5)
    plt.title("(d) Clip Fraction", fontsize=12, fontweight='bold')
    plt.xlabel("Update", fontsize=11)
    plt.ylabel("Clip Fraction", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot entropy
    plt.subplot(3, 2, 5)
    plt.plot(stats["entropies"], color='#9467bd', linewidth=1.5)
    plt.title("(e) Policy Entropy", fontsize=12, fontweight='bold')
    plt.xlabel("Update", fontsize=11)
    plt.ylabel("Entropy", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot losses with better legend placement
    plt.subplot(3, 2, 6)
    plt.plot(stats["total_losses"], label="Total Loss", color='#8c564b', linewidth=1.5)
    plt.plot(stats["policy_losses"], label="Policy Loss", color='#e377c2', linewidth=1.5)
    plt.plot(stats["value_losses"], label="Value Loss", color='#7f7f7f', linewidth=1.5)
    plt.title("(f) Training Losses", fontsize=12, fontweight='bold')
    plt.xlabel("Update", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path, bbox_inches='tight', dpi=300, format='png')
    plt.close()
    print(f"Training curve saved as {save_path}")


def main():
    parser = argparse.ArgumentParser(description="MyoSuite PPO Experiment Framework")
    
    # Basic settings
    parser.add_argument("--env-name", type=str, default="myoElbowPose1D6MRandom-v0", help="Environment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds for multi-seed training")
    
    # Experiment type
    parser.add_argument("--experiment-type", type=str, choices=["sanity", "ppo-train", "ppo-eval", "env-inspect"], 
                        default="sanity", help="Experiment type")
    
    # Sanity check parameters
    parser.add_argument("--sanity-episodes", type=int, default=20, help="Number of episodes for sanity check")
    
    # PPO training parameters
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="Total timesteps for training")
    parser.add_argument("--rollout-length", type=int, default=2048, help="Rollout length per update")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lambda-gae", type=float, default=0.95, help="GAE lambda")
    
    # Rendering parameters
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    
    # Saving parameters
    parser.add_argument("--save-model", type=str, default=None, help="Path to save the model")
    parser.add_argument("--load-model", type=str, default=None, help="Path to load the model")
    
    args = parser.parse_args()
    
    if args.experiment_type == "env-inspect":
        # Environment inspection
        manager = ExperimentManager(args.env_name)
        manager.env_inspection(render=args.render)
        manager.close()
    
    elif args.experiment_type == "sanity":
        # Three-strategy sanity check
        manager = ExperimentManager(args.env_name)
        results = manager.sanity_check(episodes=args.sanity_episodes, render=args.render)
        manager.close()
    
    elif args.experiment_type == "ppo-train":
        # PPO training
        if args.num_seeds > 1:
            # Multi-seed training
            all_stats = []
            for seed in range(args.num_seeds):
                print(f"\n=== Training with seed {seed} ===")
                env = gym.make(args.env_name)
                agent = PPOAgent(
                    env,
                    gamma=args.gamma,
                    lambda_gae=args.lambda_gae,
                    clip_range=args.clip_range,
                    lr=args.lr,
                    seed=seed
                )
                
                stats = agent.run_training(
                    total_timesteps=args.total_timesteps,
                    rollout_length=args.rollout_length,
                    render=args.render
                )
                all_stats.append(stats)
                
                if args.save_model:
                    agent.save(f"{args.save_model}_seed{seed}.pt")
                
                env.close()
            
            # Plot multi-seed statistics
            plt.figure(figsize=(15, 5))
            
            # Plot return curve with mean and standard deviation
            all_returns = [stats["returns"] for stats in all_stats]
            max_len = max(len(returns) for returns in all_returns)
            
            # Align all return curve lengths
            aligned_returns = []
            for returns in all_returns:
                if len(returns) < max_len:
                    # Fill with last value
                    padded = returns + [returns[-1]] * (max_len - len(returns))
                    aligned_returns.append(padded)
                else:
                    aligned_returns.append(returns)
            
            aligned_returns = np.array(aligned_returns)
            mean_returns = np.mean(aligned_returns, axis=0)
            std_returns = np.std(aligned_returns, axis=0)
            
            plt.plot(mean_returns, label="Mean")
            plt.fill_between(range(len(mean_returns)), mean_returns - std_returns, mean_returns + std_returns, 
                            alpha=0.3, label="Std")
            plt.title(f"PPO Training Returns (seeds={args.num_seeds})")
            plt.xlabel("Episode")
            plt.ylabel("Return")
            plt.legend()
            plt.grid(True)
            # Save to figures directory
            multi_seed_path = os.path.join("figures", f"ppo_training_returns_multi_seed_{args.env_name}.png")
            plt.savefig(multi_seed_path)
            plt.close()
            print(f"Multi-seed training returns plot saved as {multi_seed_path}")
        
        else:
            # Single-seed training
            env = gym.make(args.env_name)
            agent = PPOAgent(
                env,
                gamma=args.gamma,
                lambda_gae=args.lambda_gae,
                clip_range=args.clip_range,
                lr=args.lr,
                seed=args.seed
            )
            
            if args.load_model:
                agent.load(args.load_model)
            
            stats = agent.run_training(
                total_timesteps=args.total_timesteps,
                rollout_length=args.rollout_length,
                render=args.render
            )
            
            if args.save_model:
                agent.save(args.save_model)
            
            # Plot training curve
            plot_training_curve(stats, f"PPO Training Curve ({args.env_name})", args.env_name)
            
            # Generate value function heatmap
            agent.plot_value_heatmap(args.env_name)
            
            env.close()
    
    elif args.experiment_type == "ppo-eval":
        # PPO evaluation
        env = gym.make(args.env_name)
        agent = PPOAgent(env)
        
        if args.load_model:
            agent.load(args.load_model)
        else:
            print("Error: Must specify --load-model for evaluation")
            env.close()
            return
        
        # Run evaluation
        manager = ExperimentManager(args.env_name)
        ppo_policy = lambda obs: agent.get_action(obs)
        results = manager.run_policy(ppo_policy, episodes=args.sanity_episodes, render=args.render)
        
        # Print evaluation results
        print(f"\n=== PPO Evaluation Results ===")
        print(f"Mean Return: {np.mean(results['returns']):.3f} ± {np.std(results['returns']):.3f}")
        print(f"Mean Episode Length: {np.mean(results['episode_lengths']):.3f} ± {np.std(results['episode_lengths']):.3f}")
        print(f"Max Return: {np.max(results['returns']):.3f}")
        print(f"Min Return: {np.min(results['returns']):.3f}")
        
        manager.close()
        env.close()
    
    print("\n=== Experiment completed ===")


if __name__ == "__main__":
    main()
