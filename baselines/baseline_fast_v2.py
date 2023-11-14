import argparse
import multiprocessing
import os
import uuid
from os.path import exists
from pathlib import Path

import retro
import retro.data
from retro.enums import Actions, Observations, State
from gymnasium import Env, spaces
from red_gym_env_v2 import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

class RetroEnvWrapper(retro.RetroEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Dict({
            'obs': self.observation_space
        })

    def _update_obs(self):
        return {'obs': super()._update_obs()}
    
    def _get_obs(self):
        return {'obs': super()._get_obs()}

def make(game, state=State.DEFAULT, inttype=retro.data.Integrations.DEFAULT, **kwargs):
    """
    Create a Gym environment for the specified game
    """
    try:
        retro.data.get_romfile_path(game, inttype)
    except FileNotFoundError:
        if not retro.data.get_file_path(game, "rom.sha", inttype):
            raise
        else:
            raise FileNotFoundError(
                f"Game not found: {game}. Did you make sure to import the ROM?",
            )
    return RetroEnvWrapper(game, state, inttype=inttype, **kwargs)


def make_retro_env(game, roms_path):
    def _init():
        import retro.data
        retro.data.Integrations.add_custom_path(roms_path)
        return make(game, inttype=retro.data.Integrations.ALL, render_mode="rgb_array")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-wandb-logging", action="store_true")
    parser.add_argument("--ep-length", type=int, default=2048 * 10)
    parser.add_argument("--sess-id", default=str(uuid.uuid4())[:8])
    parser.add_argument("--sess-dir", default=os.getcwd())
    parser.add_argument("--n-envs", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--roms-path", default=os.path.join(os.getcwd(), "roms"))
    args = parser.parse_args()

    sess_path = Path(args.sess_dir) / Path(f"session_{args.sess_id}")

    env_config = {
        "headless": True,
        "save_final_state": False,
        "early_stop": False,
        "action_freq": 24,
        "init_state": "../has_pokedex_nballs.state",
        "max_steps": args.ep_length,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": "../PokemonRed.gb",
        "debug": False,
        "reward_scale": 0.5,
        "explore_weight": 2,
    }

    print(env_config)

    env = SubprocVecEnv(
        []
        + [make_env(i, env_config) for i in range(args.n_envs)]
        + [make_retro_env("Tetris-GameBoy", args.roms_path) for _ in range(1)]
        +
        # [make_retro_env('Asteroids-GameBoy', args.roms_path)] +
        # [make_retro_env('GradiusTheInterstellarAssault-GameBoy', args.roms_path)] +
        # [make_retro_env('BartSimpsonsEscapeFromCampDeadly-GameBoy', args.roms_path)]
        # [make_retro_env('Airstriker-Genesis', args.roms_path)] +
        # [make_retro_env('KirbysAdventure-Nes', args.roms_path)] +
        # [retro.make('Breakout-Atari2600', args.roms_path)] +
        # [make_retro_env('DonkeyKongCountry-Snes', args.roms_path)] +
        # [make_retro_env('SpaceInvaders-Atari2600', args.roms_path)] +
        # [make_retro_env('SonicTheHedgehog-Genesis', args.roms_path)] +
        []
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.ep_length, save_path=sess_path, name_prefix="poke"
    )

    callbacks = [checkpoint_callback] # , TensorboardCallback(sess_path)]

    if args.use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        wandb.tensorboard.patch(root_logdir=str(sess_path))
        run = wandb.init(
            project="pokemon-train",
            id=args.sess_id,
            name="less-event-log-text-test-logs-stack3-all-obs",
            config=env_config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        callbacks.append(WandbCallback())

    # env_checker.check_env(env)

    # put a checkpoint here you want to start from
    file_name = ""  # "session_9ff8e5f0/poke_21626880_steps"

    train_steps_batch = args.ep_length // 10

    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = PPO.load(file_name, env=env)
        model.n_steps = train_steps_batch
        model.n_envs = args.n_envs
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = args.n_envs
        model.rollout_buffer.reset()
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            n_steps=train_steps_batch,
            batch_size=128,
            n_epochs=1,
            gamma=0.998,
            tensorboard_log=sess_path,
        )

    print(model.policy)

    model.learn(
        total_timesteps=(args.ep_length) * args.n_envs * 10000,
        callback=CallbackList(callbacks),
        tb_log_name="poke_ppo",
    )

    if args.use_wandb_logging:
        run.finish()
