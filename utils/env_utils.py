import numpy as np
import torch
from typing import Any, Dict, Tuple
from myosuite.utils import gym

def make_env(env_name: str, **kwargs) -> gym.Env:
    """Create environment, handle environment creation logic uniformly
    
    Args:
        env_name: Environment name
        **kwargs: Environment creation parameters
        
    Returns:
        Created environment instance
    """
    return gym.make(env_name, **kwargs)

def reset_compat(env: gym.Env) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compatibility handling for environment reset, supports both Gym and Gymnasium APIs
    
    Args:
        env: Environment instance
        
    Returns:
        (obs, info) tuple
    """
    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        return reset_result  # Gymnasium: obs, info
    else:
        return reset_result, {}  # Gym: obs, empty info

def step_compat(env: gym.Env, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    """Compatibility handling for environment step, supports both Gym and Gymnasium APIs
    
    Args:
        env: Environment instance
        action: Action
        
    Returns:
        (next_obs, reward, done, info) tuple
    """
    step_result = env.step(action)
    if len(step_result) == 5:
        # Gymnasium: next_obs, reward, terminated, truncated, info
        next_obs, reward, terminated, truncated, info = step_result
        done = terminated or truncated
    else:
        # Gym: next_obs, reward, done, info
        next_obs, reward, done, info = step_result
    return next_obs, reward, done, info

def safe_close(env: gym.Env) -> None:
    """Safely close environment, handle possible exceptions
    
    Args:
        env: Environment instance
    """
    try:
        env.close()
    except Exception as e:
        print(f"Warning: Error closing environment: {e}")

def set_seed(seed: int, env: gym.Env = None) -> None:
    """Set random seed to ensure experiment reproducibility
    
    Args:
        seed: Random seed
        env: Optional environment instance
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if env is not None:
        try:
            env.seed(seed)
        except AttributeError:
            # Gymnasium uses set_random_seed instead
            if hasattr(env, 'set_random_seed'):
                env.set_random_seed(seed)
            elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'set_random_seed'):
                env.unwrapped.set_random_seed(seed)