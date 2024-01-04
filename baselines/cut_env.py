from collections import deque

from field_move_env import FieldMoveEnv


class CutEnv(FieldMoveEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, seed: int | None = None):
        self.done = False
        self.map_idxs = set()
        self.current_map_idx = 0
        self.trees_cut = 0
        self.state_queue = deque(maxlen=3)
        return super().reset(seed=seed)

    def update_obs(self):
        super().update_obs()
        self.map_idxs.add(self.read_m(0xD35E))

    def get_game_state_reward(self, print_stats: bool = False) -> dict[str, float]:
        return {
            **super().get_game_state_reward(print_stats),
            "map_idxs": self.reward_scale * len(self.map_idxs) * 0.01,
            "trees_cut": self.reward_scale * self.trees_cut * 0.1,
        }

    def check_if_done(self):
        # Arbitrarily high number of map idices
        return len(self.map_idxs) > 5

    def run_action_on_emulator(self, action):
        super().run_action_on_emulator(action)
        # 0xCFC6 - wTileInFrontOfPlayer
        # 0xCFCB - wUpdateSpritesEnabled
        self.state_queue.append(
            (self.pyboy.get_memory_value(0xCFC6), self.pyboy.get_memory_value(0xCFCB))
        )
        if self.state_queue == deque(
            [(61, 1), (61, 255), (61, 1)]
        ) or self.state_queue == deque([(80, 1), (80, 255), (80, 1)]):
            self.trees_cut += 1
        new_map_idx = self.read_m(0xD35E)
        if new_map_idx != self.current_map_idx:
            self.current_map_idx = new_map_idx
            self.state_queue.clear()
            self.trees_cut = 0
