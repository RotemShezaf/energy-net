# energy_net/env/register_envs.py

from gymnasium.envs.registration import register


print("Registering EnergyNetEnv-v0")
register(
    id='EnergyNetEnv-v0',
    entry_point='energy_net.envs.energy_net_v0:EnergyNetV0',
    # Optional parameters:
    # max_episode_steps=1000,   
    # reward_threshold=100.0,
    # nondeterministic=False,
)

# Register additional environments for RL Zoo integration

print("Registering ISO-RLZoo-v0")
register(
    id='ISO-RLZoo-v0',
    entry_point='energy_net.envs.iso_env:make_iso_env_zoo',
    max_episode_steps=48,  # Based on your config
)

print("Registering PCS-RLZoo-v0")
register(
    id='PCS-RLZoo-v0', 
    entry_point='energy_net.envs.pcs_env:make_pcs_env_zoo',
    max_episode_steps=48,  # Based on your config
)
