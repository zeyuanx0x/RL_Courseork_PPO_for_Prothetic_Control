import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from typing import Dict, List, Tuple
from myosuite.utils import gym
from utils.env_utils import reset_compat, step_compat, set_seed
from utils.render_utils import safe_render
from utils.plotting_utils import setup_plot_style, save_figure
from utils.logging_utils import setup_results_dir


class ActorCritic(nn.Module):
    """Actor-Critic Network"""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int] = [64, 64]):
        super(ActorCritic, self).__init__()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[1], act_dim),
            nn.Tanh()  # Output range [-1, 1], needs scaling later
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[1], 1)
        )
        
        # Action variance (learnable or fixed)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, MultivariateNormal]:
        """Forward pass"""
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        
        # Create action distribution
        cov_mat = torch.diag(std.pow(2))
        dist = MultivariateNormal(mean, cov_mat)
        
        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Calculate value
        value = self.critic(obs)
        
        return action, log_prob, value, dist
    
    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate probabilities, values, and entropy for given observations and actions"""
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        
        cov_mat = torch.diag(std.pow(2))
        dist = MultivariateNormal(mean, cov_mat)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(obs)
        
        return log_probs, values, entropy


class PPOAgent:
    """PPO Agent implementing PPO-clip algorithm"""
    
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        clip_range: float = 0.2,
        lr: float = 3e-4,
        num_epochs: int = 10,
        minibatch_size: int = 64,
        hidden_sizes: List[int] = [64, 64],
        seed: int = None
    ):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.act_low = env.action_space.low
        self.act_high = env.action_space.high
        
        # Hyperparameters
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_range = clip_range
        self.lr = lr
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        
        # Set random seed
        if seed is not None:
            self.set_seed(seed)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create Actor-Critic network
        self.ac = ActorCritic(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        
        # Experience buffer
        self.buffer = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "log_probs": [],
            "values": []
        }
        
        # Training statistics
        self.train_stats = {
            "returns": [],
            "episode_lengths": [],
            "kl_divs": [],
            "clip_fractions": [],
            "entropies": [],
            "value_losses": [],
            "policy_losses": [],
            "total_losses": []
        }
    
    def set_seed(self, seed: int):
        """Set random seed"""
        set_seed(seed, self.env)
    
    def scale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Scale action from [-1, 1] to environment action space"""
        # Convert NumPy arrays to PyTorch tensors
        act_low = torch.FloatTensor(self.act_low).to(action.device)
        act_high = torch.FloatTensor(self.act_high).to(action.device)
        return act_low + (action + 1.0) * 0.5 * (act_high - act_low)
    
    def collect_rollouts(self, num_steps: int, render: bool = False) -> Dict[str, List]:
        """Collect experience trajectories"""
        rollout_data = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "log_probs": [],
            "values": []
        }
        
        obs, info = reset_compat(self.env)
        episode_return = 0.0
        episode_length = 0
        
        for step in range(num_steps):
            # Sample action
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            with torch.no_grad():
                action, log_prob, value, _ = self.ac(obs_tensor)
            
            # Scale action to environment range
            scaled_action = self.scale_action(action).cpu().numpy()
            
            # Execute action
            next_obs, reward, done, info = step_compat(self.env, scaled_action)
            
            # Record data
            rollout_data["obs"].append(obs)
            rollout_data["actions"].append(action.cpu().numpy())
            rollout_data["rewards"].append(reward)
            rollout_data["dones"].append(done)
            # Ensure log_prob is a scalar
            rollout_data["log_probs"].append(log_prob.item())
            # Ensure value is a scalar
            rollout_data["values"].append(value.item())
            
            episode_return += reward
            episode_length += 1
            
            # Render
            if render:
                safe_render(self.env)
            
            # Check if episode ended
            if done:
                self.train_stats["returns"].append(episode_return)
                self.train_stats["episode_lengths"].append(episode_length)
                episode_return = 0.0
                episode_length = 0
                next_obs, info = reset_compat(self.env)
            
            obs = next_obs
        
        return rollout_data
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> np.ndarray:
        """Compute Generalized Advantage Estimation (GAE)"""
        rewards = np.array(rewards)
        values = np.array(values)
        dones = np.array(dones)
        
        advantages = np.zeros_like(rewards)
        gae = 0
        
        # Calculate GAE from back to front
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] - values[t]
            else:
                delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def train(self, rollout_data: Dict[str, List], num_epochs: int = None) -> Dict[str, float]:
        """Train the network"""
        num_epochs = num_epochs or self.num_epochs
        
        # Convert to tensors
        obs = torch.FloatTensor(rollout_data["obs"]).to(self.device)
        actions = torch.FloatTensor(rollout_data["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_data["log_probs"]).to(self.device)
        values = torch.FloatTensor(rollout_data["values"]).to(self.device)
        
        # Compute GAE and returns
        advantages, returns = self.compute_gae(
            rollout_data["rewards"],
            rollout_data["values"],
            rollout_data["dones"]
        )
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate batch sizes
        num_samples = len(obs)
        num_minibatches = num_samples // self.minibatch_size
        
        epoch_stats = {
            "kl_divs": [],
            "clip_fractions": [],
            "entropies": [],
            "value_losses": [],
            "policy_losses": [],
            "total_losses": []
        }
        
        # Train for multiple epochs
        for epoch in range(num_epochs):
            # Shuffle data
            permutation = torch.randperm(num_samples)
            obs_shuffled = obs[permutation]
            actions_shuffled = actions[permutation]
            old_log_probs_shuffled = old_log_probs[permutation]
            advantages_shuffled = advantages[permutation]
            returns_shuffled = returns[permutation]
            
            for mb_idx in range(num_minibatches):
                # Sample minibatch
                start_idx = mb_idx * self.minibatch_size
                end_idx = start_idx + self.minibatch_size
                
                mb_obs = obs_shuffled[start_idx:end_idx]
                mb_actions = actions_shuffled[start_idx:end_idx]
                mb_old_log_probs = old_log_probs_shuffled[start_idx:end_idx]
                mb_advantages = advantages_shuffled[start_idx:end_idx]
                mb_returns = returns_shuffled[start_idx:end_idx]
                
                # Evaluate actions
                log_probs, values_pred, entropy = self.ac.evaluate(mb_obs, mb_actions)
                
                # Calculate ratio
                ratio = torch.exp(log_probs - mb_old_log_probs)
                
                # Calculate policy loss
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss = nn.MSELoss()(values_pred.squeeze(), mb_returns)
                
                # Calculate entropy loss (encourage exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                
                # Backpropagate
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # Calculate KL divergence and clip fraction
                with torch.no_grad():
                    kl_div = (mb_old_log_probs - log_probs).mean().item()
                    clip_frac = (torch.abs(ratio - 1.0) > self.clip_range).float().mean().item()
                
                # Record statistics
                epoch_stats["kl_divs"].append(kl_div)
                epoch_stats["clip_fractions"].append(clip_frac)
                epoch_stats["entropies"].append(entropy.mean().item())
                epoch_stats["value_losses"].append(value_loss.item())
                epoch_stats["policy_losses"].append(policy_loss.item())
                epoch_stats["total_losses"].append(total_loss.item())
        
        # Save average statistics
        self.train_stats["kl_divs"].append(np.mean(epoch_stats["kl_divs"]))
        self.train_stats["clip_fractions"].append(np.mean(epoch_stats["clip_fractions"]))
        self.train_stats["entropies"].append(np.mean(epoch_stats["entropies"]))
        self.train_stats["value_losses"].append(np.mean(epoch_stats["value_losses"]))
        self.train_stats["policy_losses"].append(np.mean(epoch_stats["policy_losses"]))
        self.train_stats["total_losses"].append(np.mean(epoch_stats["total_losses"]))
        
        return {
            "kl_div": np.mean(epoch_stats["kl_divs"]),
            "clip_fraction": np.mean(epoch_stats["clip_fractions"]),
            "entropy": np.mean(epoch_stats["entropies"]),
            "value_loss": np.mean(epoch_stats["value_losses"]),
            "policy_loss": np.mean(epoch_stats["policy_losses"]),
            "total_loss": np.mean(epoch_stats["total_losses"])
        }
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action (for testing)"""
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            action, _, _, dist = self.ac(obs_tensor)
            if deterministic:
                action = dist.mean
        
        scaled_action = self.scale_action(action).cpu().numpy()
        return scaled_action
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            "model_state_dict": self.ac.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "hyperparams": {
                "gamma": self.gamma,
                "lambda_gae": self.lambda_gae,
                "clip_range": self.clip_range,
                "lr": self.lr,
                "num_epochs": self.num_epochs,
                "minibatch_size": self.minibatch_size
            }
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path)
        self.ac.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load hyperparameters
        hyperparams = checkpoint["hyperparams"]
        self.gamma = hyperparams["gamma"]
        self.lambda_gae = hyperparams["lambda_gae"]
        self.clip_range = hyperparams["clip_range"]
        self.lr = hyperparams["lr"]
        self.num_epochs = hyperparams["num_epochs"]
        self.minibatch_size = hyperparams["minibatch_size"]
    
    def get_stats(self) -> Dict[str, List]:
        """Get training statistics"""
        return self.train_stats
    
    def plot_value_heatmap(self, env_name: str, save_dir: str = "figures") -> None:
        """Generate value function heatmap
        
        Args:
            env_name: Environment name
            save_dir: Save directory
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        print("Generating value function heatmap...")
        
        # Get observation space info
        obs_space = self.env.observation_space
        obs_dim = obs_space.shape[0]
        obs_low = obs_space.low
        obs_high = obs_space.high
        
        # Only generate heatmap if we have at least 2 observation dimensions
        if obs_dim < 2:
            print("Skipping heatmap: Observation space has fewer than 2 dimensions")
            return
        
        # Create save directory structure using unified function
        dirs = setup_results_dir(env_name, save_dir)
        value_dir = dirs["value_heatmaps"]
        
        # Create grid for first 2 dimensions
        grid_size = 50
        dim1 = np.linspace(obs_low[0], obs_high[0], grid_size)
        dim2 = np.linspace(obs_low[1], obs_high[1], grid_size)
        
        # Create meshgrid
        X, Y = np.meshgrid(dim1, dim2)
        
        # Initialize value grid
        Z = np.zeros((grid_size, grid_size))
        
        # Evaluate value function on grid
        with torch.no_grad():
            for i in range(grid_size):
                for j in range(grid_size):
                    # Create observation with fixed values for other dimensions
                    obs = np.zeros(obs_dim)
                    obs[0] = dim1[j]  # X-axis
                    obs[1] = dim2[i]  # Y-axis
                    # Set other dimensions to their midpoint
                    for k in range(2, obs_dim):
                        obs[k] = (obs_low[k] + obs_high[k]) / 2
                    
                    # Convert to tensor and evaluate
                    obs_tensor = torch.FloatTensor(obs).to(self.device)
                    value = self.ac.critic(obs_tensor).item()
                    Z[i, j] = value
        
        # Set plot style using unified function
        setup_plot_style()
        
        # Plot heatmap
        fig = plt.figure(figsize=(12, 10))
        
        # Value heatmap
        plt.subplot(2, 1, 1)
        im = plt.imshow(Z, extent=[obs_low[0], obs_high[0], obs_low[1], obs_high[1]], 
                       origin='lower', cmap='viridis')
        plt.colorbar(im, label="Value")
        plt.title(f"Value Function Heatmap (First 2 Observation Dimensions)\n{env_name}", fontsize=14, fontweight='bold')
        plt.xlabel(f"Observation Dimension 1", fontsize=12)
        plt.ylabel(f"Observation Dimension 2", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # 3D Surface plot
        ax = plt.subplot(2, 1, 2, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
        ax.set_title(f"3D Value Function Surface\n{env_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel(f"Observation Dimension 1", fontsize=10)
        ax.set_ylabel(f"Observation Dimension 2", fontsize=10)
        ax.set_zlabel("Value", fontsize=10)
        ax.view_init(elev=30, azim=45)
        plt.colorbar(surf, label="Value", shrink=0.5, aspect=10)
        
        plt.tight_layout()
        
        # Save heatmap using unified function
        heatmap_path = save_figure(fig, f"value_heatmap_{env_name}.png", value_dir)
    
    def run_training(
        self,
        total_timesteps: int = 1e6,
        rollout_length: int = 2048,
        render: bool = False,
        log_interval: int = 1
    ) -> Dict[str, List]:
        """Run complete training process"""
        timesteps_done = 0
        update = 0
        
        while timesteps_done < total_timesteps:
            # Collect experience
            rollout_data = self.collect_rollouts(rollout_length, render=render)
            timesteps_done += len(rollout_data["obs"])
            
            # Train network
            train_stats = self.train(rollout_data)
            update += 1
            
            # Record and print statistics
            if update % log_interval == 0:
                if self.train_stats["returns"]:
                    avg_return = np.mean(self.train_stats["returns"][-10:])
                    avg_length = np.mean(self.train_stats["episode_lengths"][-10:])
                else:
                    avg_return = 0.0
                    avg_length = 0.0
                
                print(f"Update {update} | Timesteps: {timesteps_done:.0f} | Avg Return: {avg_return:.2f} | Avg Length: {avg_length:.1f}")
                print(f"  KL: {train_stats['kl_div']:.4f} | Clip: {train_stats['clip_fraction']:.3f} | Entropy: {train_stats['entropy']:.3f}")
                print(f"  Value Loss: {train_stats['value_loss']:.4f} | Policy Loss: {train_stats['policy_loss']:.4f}")
        
        return self.get_stats()
