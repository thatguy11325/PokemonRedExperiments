from field_move_env import FieldMoveEnv


class CutEnv(FieldMoveEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.done = False
        self.map_idxs = set()

    def update_obs(self):
        super().update_obs()
        self.map_idxs.add(self.read_m(0xD35E))

    def get_game_state_reward(self, print_stats: bool=False) -> dict[str, float]:
        return {
            **super().get_game_state_reward(print_stats),
            "map_idxs": self.reward_scale * len(self.map_idxs) * 0.1
        }

    def check_if_done(self):
        # Arbitrarily high number of map idices
        return len(self.map_idxs) > 5
