"""
PCS Environment Factory for RL-Baselines3-Zoo integration.

This module provides factory functions that create PCS-focused environments
wrapped appropriately for training with RL-Baselines3-Zoo.
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.monitor import Monitor

from energy_net.envs import EnergyNetV0


def make_pcs_env_zoo(
    norm_path=None,
    iso_policy_path=None,
    log_dir="logs",
    use_dispatch_action=False,
    dispatch_strategy="PROPORTIONAL",
    iso_policy_hyperparams_path=None,
    monitor=True,
    seed=None,
    **kwargs
):
    """
    Factory function for PCS environment compatible with RL-Baselines3-Zoo.
    
    Args:
        norm_path: Path to saved normalization statistics
        iso_policy_path: Path to a trained ISO policy to use during PCS training
        log_dir: Directory for saving logs
        use_dispatch_action: Whether to include dispatch in ISO action space
        dispatch_strategy: Strategy for dispatch when not controlled by agent
        monitor: Whether to wrap with Monitor for episode stats
        seed: Random seed
        **kwargs: Additional arguments to pass to EnergyNetV0
        
    Returns:
        A PCS-focused environment ready for training with RL-Baselines3-Zoo
    """
    # Create monitor directory if it doesn't exist
    if monitor:
        monitor_dir = os.path.join(log_dir, "pcs_monitor")
        os.makedirs(monitor_dir, exist_ok=True)
    
    # Create base environment
    env_kwargs = {
        "dispatch_config": {
            "use_dispatch_action": use_dispatch_action,
            "default_strategy": dispatch_strategy
        }
    }
    env_kwargs.update(kwargs)
    
    env = EnergyNetV0(**env_kwargs)

    if iso_policy_path:
        # Load ISO policy if provided
        iso_policy = None
        import yaml
        try:
            # Load hyperparameters from YAML
            learning_rate = 1e-4  # Default fallback
            if iso_policy_hyperparams_path:
                with open(iso_policy_hyperparams_path, 'r') as f:
                    params = yaml.safe_load(f)


                # Extract and convert learning_rate if present
                if 'PCS-RLZoo-v0' in params and 'learning_rate' in params['PCS-RLZoo-v0']:
                    lr_val = params['PCS-RLZoo-v0']['learning_rate']
                    if isinstance(lr_val, str) and lr_val.startswith('lin_'):
                        learning_rate = float(lr_val.replace('lin_', ''))
                    elif isinstance(lr_val, (float, int)):
                        learning_rate = lr_val
            custom_objects = {'learning_rate': learning_rate}
            print(f"Loading ISO policy from {iso_policy_path}")
            # Determine algorithm type from path
            breakpoint()
            if 'td3' in iso_policy_path.lower():

                iso_policy = TD3.load(iso_policy_path, custom_objects= custom_object)
                print("Loaded TD3 ISO policy")
            else:
                iso_policy = PPO.load(iso_policy_path, custom_objects= custom_objects)
                print("Loaded PPO ISO policy")
        except Exception as e:
            print(f"Error loading ISO policy: {e}")
    
    # Import here to avoid circular imports
    from energy_net.controllers.alternating_wrappers import PCSEnvWrapper
    
    # Apply PCS wrapper
    env = PCSEnvWrapper(env, iso_policy=iso_policy)
    
    # Apply monitor wrapper if requested
    if monitor:
        env = Monitor(env, monitor_dir, allow_early_resets=True)
    
    # Set random seed if provided
    if seed is not None:
        env.reset(seed = seed)
    
    return env 
