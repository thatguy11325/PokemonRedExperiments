from hm_env import HmEnv


class SurfEnv(HmEnv):
    def check_if_done(self) -> float:
        # https://github.com/pret/pokered/blob/master/ram/wram.asm#L2052
        print(self.read_m(0xD700))
        return self.read_m(0xD700) == 0x02
