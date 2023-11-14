import os
import uuid
from os.path import exists
from pathlib import Path

import retro
from red_gym_env_v2 import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback

def make_retro_env(game, roms_path):
    def _init():
        retro.data.Integrations.add_custom_path(roms_path)
        return retro.make(game, inttype=retro.data.Integrations.ALL, render_mode="rgb_array")
    return _init



def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':

    use_wandb_logging = False
    ep_length = 2048 * 10
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_{sess_id}')

    env_config = {
                'headless': True, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 
                'use_screen_explore': True, 'reward_scale': 4, 'extra_buttons': False,
                'explore_weight': 3 # 2.5
            }
    
    print(env_config)
    
    roms_path = os.path.join(os.getcwd(), "roms")
    num_cpu = 4  # Also sets the number of episodes per training iteration
    # print(retro.data.list_games(inttype=retro.data.Integrations.ALL))
    env = SubprocVecEnv(
        [make_env(i, env_config) for i in range(4)] +
        # [make_retro_env('Tetris-GameBoy', roms_path)] +
        # [make_retro_env('Asteroids-GameBoy', roms_path)] +
        # [make_retro_env('GradiusTheInterstellarAssault-GameBoy', roms_path)] +
        # [make_retro_env('BartSimpsonsEscapeFromCampDeadly-GameBoy', roms_path)]
        # [make_retro_env('Airstriker-Genesis', roms_path)] +
        # [make_retro_env('KirbysAdventure-Nes', roms_path)] +
        # [retro.make('Breakout-Atari2600', roms_path)] +
        # [make_retro_env('DonkeyKongCountry-Snes', roms_path)] +
        # [make_retro_env('SpaceInvaders-Atari2600', roms_path)] +
        # [make_retro_env('SonicTheHedgehog-Genesis', roms_path)] +
        []
    )
    
    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                     name_prefix='poke')
    
    callbacks = [checkpoint_callback, TensorboardCallback(log_dir="./logs")]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            config=env_config,
            sync_tensorboard=True,  
            monitor_gym=True,  
            save_code=True,
        )
        callbacks.append(WandbCallback())

    #env_checker.check_env(env)
    learn_steps = 40
    # put a checkpoint here you want to start from
    file_name = 'session_e41c9eff/poke_38207488_steps' 
    
    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = env.num_envs
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length // 8, batch_size=128, n_epochs=3, gamma=0.998, tensorboard_log=sess_path)
    
    for i in range(learn_steps):
        model.learn(total_timesteps=(ep_length)*num_cpu*1000, callback=CallbackList(callbacks))

    if use_wandb_logging:
        run.finish()
