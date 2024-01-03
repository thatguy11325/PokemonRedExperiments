from field_move_env import FieldMoveEnv


class SurfEnv(FieldMoveEnv):
    def get_game_state_reward(self, print_stats: bool=False) -> dict[str, float]:
        return {
            **super().get_game_state_reward(print_stats),
            "surf": float(self.read_m(0xD700) == 0x02),
        }

    def check_if_done(self) -> float:
        # https://github.com/pret/pokered/blob/master/ram/wram.asm#L2052
        return self.read_m(0xD700) == 0x02
