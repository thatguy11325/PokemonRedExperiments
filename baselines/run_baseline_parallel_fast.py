import argparse
import multiprocessing
import os
import random
import uuid
from os.path import exists
from pathlib import Path

import torch
import torchvision
from cut_env import CutEnv
from red_gym_env_v2 import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from surf_env import SurfEnv
from tensorboard_callback import TensorboardCallback
from warp_env import WarpEnv


def make_env(rank, env_type, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        if env_type == "surf":
            env_class = SurfEnv
        elif env_type == "cut":
            env_class = CutEnv
        elif env_type == "warp":
            env_class = WarpEnv
        else:
            env_class = RedGymEnv
        env = env_class(env_conf)
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rom-path", default="../PokemonRed.gb")
    parser.add_argument("--state-path", default="../home.state")
    parser.add_argument("--n-envs", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--wandb-api-key", type=str, default=None)
    parser.add_argument("--ep-length", type=int, default=2048 * 10)
    parser.add_argument("--sess-id", type=str, default=str(uuid.uuid4())[:8])
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--fast-video", action="store_true")
    parser.add_argument("--frame-stacks", type=int, default=4)
    parser.add_argument(
        "--policy",
        choices=["MultiInputPolicy", "CnnPolicy"],
        default="MultiInputPolicy2",
    )
    parser.add_argument(
        "--vec-env-type", choices=["subproc", "dummy"], default="subproc"
    )
    parser.add_argument(
        "--poke-env-type",
        type=str,
        choices=["surf", "cut", "warp", "all"],
        default="all",
    )
    parser.add_argument("--device", type=str, default=detect_device())
    parser.add_argument(
        "--seed-style",
        type=str,
        choices=["random", "buckets"],
        default="random",
        help="Random seed is every env starts with a random delay. "
        "Bucketed is every 4 envs start 4096 steps delayed from the previous 4 envs.",
    )

    args = parser.parse_args()

    sess_path = Path(f"session_{args.sess_id}")

    env_config = {
        "headless": args.headless,
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": args.state_path,
        "max_steps": args.ep_length,
        "print_rewards": True,
        "save_video": args.save_video,
        "fast_video": args.fast_video,
        "session_path": sess_path,
        "gb_path": args.rom_path,
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "use_screen_explore": True,
        "reward_scale": 4,
        "extra_buttons": False,
        "explore_weight": 3,  # 2.5
        "explore_npc_weight": 5,  # 2.5
        "frame_stacks": args.frame_stacks,
        "policy": args.policy,
    }

    print(env_config)

    roms_path = os.path.join(os.getcwd(), "roms")
    vecenv_type = SubprocVecEnv if args.vec_env_type == "subproc" else DummyVecEnv
    env = vecenv_type(
        [
            make_env(
                i,
                args.poke_env_type,
                env_config,
                seed=random.randint(0, 4096)
                if args.seed_style == "random"
                else 4096 * i // 4,
            )
            for i in range(args.n_envs)
        ]
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.ep_length, save_path=sess_path, name_prefix="poke"
    )

    callbacks = [checkpoint_callback, TensorboardCallback(log_dir="./logs")]

    if args.wandb_api_key:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        run = wandb.init(
            project="pokemon-train",
            id=args.sess_id,
            config=env_config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        callbacks.append(WandbCallback())

    # env_checker.check_env(env)
    learn_steps = 40
    # put a checkpoint here you want to start from
    file_name = "session_e41c9eff/poke_38207488_steps"

    policy_kwargs = None
    if args.policy == "CnnPolicy":
        policy_kwargs = dict(
            features_extractor_class=torchvision.models.resnet152,
            features_extractor_kwargs=dict(pretrained=True),
        )
    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = PPO.load(file_name, env=env)
        model.n_steps = args.ep_length
        model.n_envs = env.num_envs
        model.rollout_buffer.buffer_size = args.ep_length
        model.rollout_buffer.n_envs = args.n_envs
        model.rollout_buffer.reset()
    else:
        model = PPO(
            args.policy,
            env,
            verbose=1,
            n_steps=args.ep_length // 8,
            batch_size=128,
            n_epochs=3,
            gamma=0.998,
            tensorboard_log=sess_path,
            device=args.device,
        )

    if args.device == "cuda":
        model.policy = torch.compile(model.policy, mode="max-autotune")
    for i in range(learn_steps):
        model.learn(
            total_timesteps=(args.ep_length) * args.n_envs * 1000,
            callback=CallbackList(callbacks),
        )

    if args.use_wandb_logging:
        run.finish()
