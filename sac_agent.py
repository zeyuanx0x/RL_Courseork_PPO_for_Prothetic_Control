import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple
from myosuite.utils import gym
from utils.env_utils import reset_compat, step_compat, set_seed


class Actor(nn.Module):
    """SAC Actor Network"""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int] = [256, 256]):
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], act_dim * 2)  # Mean and log_std
        )
        
        # Set log_std bounds
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mu, log_std = torch.chunk(self.net(obs), 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action with reparameterization trick"""
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        
        if deterministic:
            u = mu
        else:
            u = Normal(0, 1).sample(mu.shape)
        
        # Reparameterization trick
        action = torch.tanh(mu + u * std)
        
        # Calculate log probability
        log_prob = Normal(mu, std).log_prob(mu + u * std)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """SAC Critic Network"""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int] = [256, 256]):
        super(Critic, self).__init__()
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )
        
        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        x = torch.cat([obs, action], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2


class SACAgent:
    """Soft Actor-Critic Agent"""
    
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        lr: float = 3e-4,
        hidden_sizes: List[int] = [256, 256],
        seed: int = None
    ):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.act_low = env.action_space.low
        self.act_high = env.action_space.high
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lr = lr
        
        # Set random seed
        if seed is not None:
            self.set_seed(seed)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create networks
        self.actor = Actor(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        self.critic = Critic(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        self.target_critic = Critic(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        
        # Initialize target critic weights
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Experience buffer
        self.buffer = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "next_obs": []
        }
        self.buffer_size = 1000000
        self.batch_size = 256
        
        # Training statistics
        self.train_stats = {
            "returns": [],
            "episode_lengths": [],
            "actor_losses": [],
            "critic_losses": [],
            "entropies": []
        }
    
    def set_seed(self, seed: int):
        """Set random seed"""
        set_seed(seed, self.env)
    
    def scale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Scale action from [-1, 1] to environment action space"""
        act_low = torch.FloatTensor(self.act_low).to(action.device)
        act_high = torch.FloatTensor(self.act_high).to(action.device)
        return act_low + (action + 1.0) * 0.5 * (act_high - act_low)
    
    def collect_rollouts(self, num_steps: int, render: bool = False) -> None:
        """Collect experience trajectories"""
        obs, info = reset_compat(self.env)
        episode_return = 0.0
        episode_length = 0
        
        for step in range(num_steps):
            # Sample action
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            with torch.no_grad():
                action, _ = self.actor.sample(obs_tensor)
            
            # Scale action to environment range
            scaled_action = self.scale_action(action).cpu().numpy()
            
            # Execute action
            next_obs, reward, done, info = step_compat(self.env, scaled_action)
            
            # Store experience
            if len(self.buffer["obs"]) >= self.buffer_size:
                # Remove oldest experience
                for key in self.buffer:
                    self.buffer[key].pop(0)
            
            self.buffer["obs"].append(obs)
            self.buffer["actions"].append(action.cpu().numpy())
            self.buffer["rewards"].append(reward)
            self.buffer["dones"].append(done)
            self.buffer["next_obs"].append(next_obs)
            
            episode_return += reward
            episode_length += 1
            
            # Check if episode ended
            if done:
                self.train_stats["returns"].append(episode_return)
                self.train_stats["episode_lengths"].append(episode_length)
                episode_return = 0.0
                episode_length = 0
                next_obs, info = reset_compat(self.env)
            
            obs = next_obs
    
    def train(self, num_updates: int) -> Dict[str, float]:
        """Train the network"""
        if len(self.buffer["obs"]) < self.batch_size:
            return {}
        
        epoch_stats = {
            "actor_losses": [],
            "critic_losses": [],
            "entropies": []
        }
        
        for _ in range(num_updates):
            # Sample batch
            batch_indices = np.random.randint(0, len(self.buffer["obs"]), size=self.batch_size)
            
            obs_batch = torch.FloatTensor([self.buffer["obs"][i] for i in batch_indices]).to(self.device)
            actions_batch = torch.FloatTensor([self.buffer["actions"][i] for i in batch_indices]).to(self.device)
            rewards_batch = torch.FloatTensor([self.buffer["rewards"][i] for i in batch_indices]).to(self.device).unsqueeze(1)
            dones_batch = torch.FloatTensor([self.buffer["dones"][i] for i in batch_indices]).to(self.device).unsqueeze(1)
            next_obs_batch = torch.FloatTensor([self.buffer["next_obs"][i] for i in batch_indices]).to(self.device)
            
            # Get target actions and log probs for next observations
            next_actions, next_log_probs = self.actor.sample(next_obs_batch)
            
            # Get target Q values
            with torch.no_grad():
                target_q1, target_q2 = self.target_critic(next_obs_batch, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
                target_q = rewards_batch + (1 - dones_batch) * self.gamma * target_q
            
            # Get current Q values
            current_q1, current_q2 = self.critic(obs_batch, actions_batch)
            
            # Calculate critic loss
            critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Calculate actor loss
            new_actions, log_probs = self.actor.sample(obs_batch)
            q1, q2 = self.critic(obs_batch, new_actions)
            q = torch.min(q1, q2)
            actor_loss = (self.alpha * log_probs - q).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target critic
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            # Record statistics
            epoch_stats["actor_losses"].append(actor_loss.item())
            epoch_stats["critic_losses"].append(critic_loss.item())
            epoch_stats["entropies"].append(-log_probs.mean().item())
        
        # Save average statistics
        self.train_stats["actor_losses"].append(np.mean(epoch_stats["actor_losses"]))
        self.train_stats["critic_losses"].append(np.mean(epoch_stats["critic_losses"]))
        self.train_stats["entropies"].append(np.mean(epoch_stats["entropies"]))
        
        return {
            "actor_loss": np.mean(epoch_stats["actor_losses"]),
            "critic_loss": np.mean(epoch_stats["critic_losses"]),
            "entropy": np.mean(epoch_stats["entropies"])
        }
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action (for testing)"""
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            action, _ = self.actor.sample(obs_tensor, deterministic=deterministic)
        
        scaled_action = self.scale_action(action).cpu().numpy()
        return scaled_action
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "hyperparams": {
                "gamma": self.gamma,
                "tau": self.tau,
                "alpha": self.alpha,
                "lr": self.lr
            }
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        
        # Load hyperparameters
        hyperparams = checkpoint["hyperparams"]
        self.gamma = hyperparams["gamma"]
        self.tau = hyperparams["tau"]
        self.alpha = hyperparams["alpha"]
        self.lr = hyperparams["lr"]
    
    def get_stats(self) -> Dict[str, List]:
        """Get training statistics"""
        return self.train_stats
    
    def run_training(
        self,
        total_timesteps: int = 1e6,
        steps_per_update: int = 1,
        updates_per_step: int = 1,
        render: bool = False,
        log_interval: int = 1
    ) -> Dict[str, List]:
        """Run complete training process"""
        timesteps_done = 0
        update = 0
        
        while timesteps_done < total_timesteps:
            # Collect experience
            self.collect_rollouts(steps_per_update, render=render)
            timesteps_done += steps_per_update
            
            # Train network
            train_stats = self.train(updates_per_step)
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
                if train_stats:
                    print(f"  Actor Loss: {train_stats['actor_loss']:.4f} | Critic Loss: {train_stats['critic_loss']:.4f} | Entropy: {train_stats['entropy']:.3f}")
        
        return self.get_stats()