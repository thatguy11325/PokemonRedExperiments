import json
import uuid
from pathlib import Path
from typing import Any

import mediapy as media
import numpy as np
from einops import repeat
from gymnasium import Env, spaces
from pyboy import PyBoy, WindowEvent
from skimage.transform import downscale_local_mean


class FieldMoveEnv(Env):
    def __init__(self, config: dict[str, int | float] | None = None):
        self.s_path: Path = config["session_path"]
        self.save_final_state = config["save_final_state"]
        self.print_rewards = config["print_rewards"]
        self.headless = config["headless"]
        self.init_state = config["init_state"]
        self.act_freq = config["action_freq"]
        self.max_steps = config["max_steps"]
        self.save_video = config["save_video"]
        self.fast_video = config["fast_video"]
        self.frame_stacks = config["frame_stacks"]
        self.policy = config["policy"]
        self.explore_weight = (
            1 if "explore_weight" not in config else config["explore_weight"]
        )
        self.reward_scale = (
            1 if "reward_scale" not in config else config["reward_scale"]
        )
        self.policy = config["policy"]
        self.instance_id = (
            str(uuid.uuid4())[:8]
            if "instance_id" not in config
            else config["instance_id"]
        )
        self.s_path.mkdir(exist_ok=True)
        self.full_frame_writer = None
        self.model_frame_writer = None
        self.map_frame_writer = None
        self.reset_count = 0
        self.step_count = 0
        self.all_runs = []

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
        ]

        # load event names (parsed from https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm)
        with open("events.json") as f:
            event_names = json.load(f)
        self.event_names = event_names

        self.output_shape = (72, 80, self.frame_stacks)
        self.coords_pad = 12

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))

        self.enc_freqs = 8

        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.output_shape, dtype=np.uint8
        )

        head = "headless" if config["headless"] else "SDL2"

        self.pyboy = PyBoy(
            config["gb_path"],
            debugging=False,
            disable_input=False,
            window_type=head,
        )

        self.screen = self.pyboy.botsupport_manager().screen()

        if not config["headless"]:
            self.pyboy.set_emulation_speed(6)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[Any, Any]]:
        self.seed = seed

        # restart game, skipping creditsw]
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        # lazy random seed setting
        if self.seed:
            for _ in range(self.seed):
                self.pyboy.tick()

        self.init_map_mem()

        self.agent_stats = []

        self.explore_map_dim = 384
        self.explore_map = np.zeros(
            (self.explore_map_dim, self.explore_map_dim), dtype=np.uint8
        )

        self.recent_screens = np.zeros(self.output_shape, dtype=np.uint8)
        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        return self._get_obs(), {}

    def init_map_mem(self):
        self.seen_coords = {}

    def render(self, reduce_res: bool = True) -> np.ndarray:
        game_pixels_render = self.screen.screen_ndarray()[:, :, 0:1]  # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (
                downscale_local_mean(game_pixels_render, (2, 2, 1))
            ).astype(np.uint8)
        return game_pixels_render

    def update_recent_screens(self, cur_screen):
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = cur_screen[:, :, 0]

    def _get_obs(self) -> np.ndarray:
        screen = self.render()
        self.update_recent_screens(screen)
        return self.recent_screens

    def read_m(self, addr: int) -> int:
        return self.pyboy.get_memory_value(addr)

    def get_game_coords(self) -> tuple[int, int, int]:
        return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

    def update_seen_coords(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        self.seen_coords[(x_pos, y_pos, map_n)] = self.step_count

    def get_global_coords(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        c = (
            np.array([x_pos, -y_pos])
            + self.get_map_location(map_n)["coordinates"]
            + self.coords_pad * 2
        )
        return self.explore_map.shape[0] - c[1], c[0]

    def update_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            # print(f"coord out of bounds! global: {c} game: {self.get_game_coords()}")
            pass
        else:
            self.explore_map[c[0], c[1]] = 255

    def get_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            out = np.zeros((self.coords_pad * 2, self.coords_pad * 2), dtype=np.uint8)
        else:
            out = self.explore_map[
                c[0] - self.coords_pad : c[0] + self.coords_pad,
                c[1] - self.coords_pad : c[1] + self.coords_pad,
            ]
        return repeat(out, "h w -> (h h2) (w w2)", h2=2, w2=2)

    def get_game_state_reward(self, print_stats: bool=False) -> dict[str, float]:
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        return {
            "explore": self.reward_scale * self.explore_weight * len(self.seen_coords) * 0.01,
        }

    def update_reward(self):
        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum([val for _, val in self.progress_reward.items()])
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return new_step
    
    def check_if_done(self) -> float:
        raise NotImplementedError("Must be implemented in subclass")

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[Any, Any]]:
        if self.save_video and self.step_count == 0:
            self.start_video()

        self.run_action_on_emulator(action)
        self.append_agent_stats(action)
        self.update_obs()
        new_reward = self.update_reward()
        step_limit_reached = self.check_if_done()
        obs = self._get_obs()

        self.step_count += 1

        return obs, new_reward, False, step_limit_reached, {}

    def update_obs(self):
        self.update_seen_coords()
        self.update_explore_map()


    def run_action_on_emulator(self, action: int):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if not self.save_video and self.headless:
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8 and action < len(self.release_actions):
                # release button
                self.pyboy.send_input(self.release_actions[action])

            if self.save_video and not self.fast_video:
                self.add_video_frame()
            if i == self.act_freq - 1:
                # rendering must be enabled on the tick before frame is needed
                self.pyboy._rendering(True)
            self.pyboy.tick()

        if self.save_video and self.fast_video:
            self.add_video_frame()

    def append_agent_stats(self, action: int):
        x_pos, y_pos, map_n = self.get_game_coords()
        self.agent_stats.append(
            {
                "step": self.step_count,
                "x": x_pos,
                "y": y_pos,
                "map": map_n,
                "map_location": self.get_map_location(map_n),
                "max_map_progress": self.max_map_progress,
                "last_action": action,
                "coord_count": len(self.seen_coords),
            }
        )

    def start_video(self):
        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()
        if self.map_frame_writer is not None:
            self.map_frame_writer.close()

        base_dir = self.s_path / Path("rollouts")
        base_dir.mkdir(exist_ok=True)
        full_name = Path(
            f"full_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        model_name = Path(
            f"model_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.full_frame_writer = media.VideoWriter(
            base_dir / full_name, (144, 160), fps=60, input_format="gray"
        )
        self.full_frame_writer.__enter__()
        self.model_frame_writer = media.VideoWriter(
            base_dir / model_name, self.output_shape[:2], fps=60, input_format="gray"
        )
        self.model_frame_writer.__enter__()
        map_name = Path(
            f"map_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.map_frame_writer = media.VideoWriter(
            base_dir / map_name,
            (self.coords_pad * 4, self.coords_pad * 4),
            fps=60,
            input_format="gray",
        )
        self.map_frame_writer.__enter__()

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render(reduce_res=False)[:, :, 0])
        self.model_frame_writer.add_image(self.render(reduce_res=True)[:, :, 0])
        self.map_frame_writer.add_image(self.get_explore_map())

    def get_map_location(self, map_idx: int) -> dict[str, Any]:
        map_locations = {
            0: {"name": "Pallet Town", "coordinates": np.array([70, 7])},
            1: {"name": "Viridian City", "coordinates": np.array([60, 79])},
            2: {"name": "Pewter City", "coordinates": np.array([60, 187])},
            3: {"name": "Cerulean City", "coordinates": np.array([240, 205])},
            62: {
                "name": "Invaded house (Cerulean City)",
                "coordinates": np.array([290, 227]),
            },
            63: {
                "name": "trade house (Cerulean City)",
                "coordinates": np.array([290, 212]),
            },
            64: {
                "name": "Pokémon Center (Cerulean City)",
                "coordinates": np.array([290, 197]),
            },
            65: {
                "name": "Pokémon Gym (Cerulean City)",
                "coordinates": np.array([290, 182]),
            },
            66: {
                "name": "Bike Shop (Cerulean City)",
                "coordinates": np.array([290, 167]),
            },
            67: {
                "name": "Poké Mart (Cerulean City)",
                "coordinates": np.array([290, 152]),
            },
            35: {"name": "Route 24", "coordinates": np.array([250, 235])},
            36: {"name": "Route 25", "coordinates": np.array([270, 267])},
            12: {"name": "Route 1", "coordinates": np.array([70, 43])},
            13: {"name": "Route 2", "coordinates": np.array([70, 151])},
            14: {"name": "Route 3", "coordinates": np.array([100, 179])},
            15: {"name": "Route 4", "coordinates": np.array([150, 197])},
            33: {"name": "Route 22", "coordinates": np.array([20, 71])},
            37: {"name": "Red house first", "coordinates": np.array([61, 9])},
            38: {"name": "Red house second", "coordinates": np.array([61, 0])},
            39: {"name": "Blues house", "coordinates": np.array([91, 9])},
            40: {"name": "oaks lab", "coordinates": np.array([91, 1])},
            41: {
                "name": "Pokémon Center (Viridian City)",
                "coordinates": np.array([100, 54]),
            },
            42: {
                "name": "Poké Mart (Viridian City)",
                "coordinates": np.array([100, 62]),
            },
            43: {"name": "School (Viridian City)", "coordinates": np.array([100, 79])},
            44: {"name": "House 1 (Viridian City)", "coordinates": np.array([100, 71])},
            47: {
                "name": "Gate (Viridian City/Pewter City) (Route 2)",
                "coordinates": np.array([91, 143]),
            },
            49: {"name": "Gate (Route 2)", "coordinates": np.array([91, 115])},
            50: {
                "name": "Gate (Route 2/Viridian Forest) (Route 2)",
                "coordinates": np.array([91, 115]),
            },
            51: {"name": "viridian forest", "coordinates": np.array([35, 144])},
            52: {"name": "Pewter Museum (floor 1)", "coordinates": np.array([60, 196])},
            53: {"name": "Pewter Museum (floor 2)", "coordinates": np.array([60, 205])},
            54: {
                "name": "Pokémon Gym (Pewter City)",
                "coordinates": np.array([49, 176]),
            },
            55: {
                "name": "House with disobedient Nidoran♂ (Pewter City)",
                "coordinates": np.array([51, 184]),
            },
            56: {"name": "Poké Mart (Pewter City)", "coordinates": np.array([40, 170])},
            57: {
                "name": "House with two Trainers (Pewter City)",
                "coordinates": np.array([51, 184]),
            },
            58: {
                "name": "Pokémon Center (Pewter City)",
                "coordinates": np.array([45, 161]),
            },
            59: {
                "name": "Mt. Moon (Route 3 entrance)",
                "coordinates": np.array([153, 234]),
            },
            60: {"name": "Mt. Moon Corridors", "coordinates": np.array([168, 253])},
            61: {"name": "Mt. Moon Level 2", "coordinates": np.array([197, 253])},
            68: {
                "name": "Pokémon Center (Route 3)",
                "coordinates": np.array([135, 197]),
            },
            193: {
                "name": "Badges check gate (Route 22)",
                "coordinates": np.array([0, 87]),
            },  # TODO this coord is guessed, needs to be updated
            230: {
                "name": "Badge Man House (Cerulean City)",
                "coordinates": np.array([290, 137]),
            },
        }
        if map_idx in map_locations.keys():
            return map_locations[map_idx]
        else:
            return {
                "name": "Unknown",
                "coordinates": np.array([80, 0]),
            }  # TODO once all maps are added this case won't be needed
