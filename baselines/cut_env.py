from hm_env import HmEnv


class CutEnv(HmEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.done = False
        self.map_idxs = set()

    def update_obs(self):
        super().update_obs()
        self.map_idxs.add(self.read_m(0xD35E))

    def check_if_done(self):
        # Arbitrarily high number of map idices
        return len(self.map_idxs) > 5
