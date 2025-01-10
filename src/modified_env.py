from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv

import numpy as np


class ModifiedHumanoidEnv(HumanoidEnv):
    def __init__(self, list_of_obs=[]):
        super().__init__()
        self.list_of_obs = list_of_obs

    def _get_obs(self):
        raw_obs = super()._get_obs()
        # print(raw_obs)
        # # print(return_val)
        # # return return_val[:100]
        return_vect = np.zeros(348)

        if "position" in self.list_of_obs:
            return_vect[0:22] = raw_obs[0:22]

        if "velocity" in self.list_of_obs:
            return_vect[22:45] = raw_obs[22:45]

        if "com_inertia" in self.list_of_obs:
            return_vect[45:170] = raw_obs[45:170]

        if "com_velocity" in self.list_of_obs:
            return_vect[170:253] = raw_obs[170:253]

        if "actuator_forces" in self.list_of_obs:
            return_vect[253:270] = raw_obs[253:270]

        if "external_contact_forces" in self.list_of_obs:
            return_vect[270:348] = raw_obs[270:348]

        # print(return_vect)
        
        return return_vect