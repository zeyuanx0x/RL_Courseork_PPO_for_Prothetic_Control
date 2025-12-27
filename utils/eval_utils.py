import numpy as np
from typing import Callable, Dict, List, Any
from myosuite.utils import gym
from utils.env_utils import reset_compat, step_compat

def run_policy(
    env: gym.Env,
    policy: Callable,
    episodes: int = 10,
    render: bool = False,
    render_func: Callable = None
) -> Dict[str, List]:
    """Run policy and collect data
    
    Args:
        env: Environment instance
        policy: Policy function, takes observation and returns action
        episodes: Number of test episodes
        render: Whether to render
        render_func: Custom rendering function
        
    Returns:
        Dictionary containing returns, episode lengths, etc.
    """
    results = {
        "returns": [],
        "episode_lengths": [],
        "all_rewards": [],
        "all_actions": [],
        "all_obs": []
    }
    
    for ep in range(episodes):
        obs, info = reset_compat(env)
        ep_return = 0.0
        ep_steps = 0
        
        ep_rewards = []
        ep_actions = []
        ep_obs = []
        
        while True:
            action = policy(obs)
            next_obs, reward, done, info = step_compat(env, action)
            
            ep_return += reward
            ep_steps += 1
            
            ep_rewards.append(reward)
            ep_actions.append(action)
            ep_obs.append(obs)
            
            obs = next_obs
            
            # Render
            if render:
                if render_func:
                    render_func(env)
                else:
                    env.render()
            
            if done:
                break
        
        results["returns"].append(ep_return)
        results["episode_lengths"].append(ep_steps)
        results["all_rewards"].append(ep_rewards)
        results["all_actions"].append(ep_actions)
        results["all_obs"].append(ep_obs)
    
    return results

def collect_data(
    env: gym.Env,
    policy: Callable,
    max_steps: int = 1000,
    render: bool = False,
    render_func: Callable = None
) -> Dict[str, List]:
    """Collect single episode data
    
    Args:
        env: Environment instance
        policy: Policy function
        max_steps: Maximum number of steps
        render: Whether to render
        render_func: Custom rendering function
        
    Returns:
        Dictionary containing episode data
    """
    obs, info = reset_compat(env)
    
    data = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "info": [],
        "return": 0.0,
        "length": 0
    }
    
    for step in range(max_steps):
        action = policy(obs)
        next_obs, reward, done, info = step_compat(env, action)
        
        data["obs"].append(obs)
        data["actions"].append(action)
        data["rewards"].append(reward)
        data["dones"].append(done)
        data["info"].append(info)
        data["return"] += reward
        data["length"] = step + 1
        
        obs = next_obs
        
        # Render
        if render:
            if render_func:
                render_func(env)
            else:
                env.render()
        
        if done:
            break
    
    return data

def calculate_statistics(data: List[float], name: str = "data") -> Dict[str, float]:
    """Calculate statistics
    
    Args:
        data: Data list
        name: Data name
        
    Returns:
        Dictionary containing mean, std, min, max, etc.
    """
    return {
        f"avg_{name}": np.mean(data),
        f"std_{name}": np.std(data),
        f"min_{name}": np.min(data),
        f"max_{name}": np.max(data),
        f"median_{name}": np.median(data)
    }

def print_statistics(stats: Dict[str, float], prefix: str = ""):
    """Print statistics
    
    Args:
        stats: Statistics dictionary
        prefix: Print prefix
    """
    for key, value in stats.items():
        print(f"  {prefix}{key}: {value:.3f}")