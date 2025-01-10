import gymnasium as gym

from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
from stable_baselines3.common.logger import configure


from stable_baselines3 import PPO
import numpy as np


tmp_path = "/tmp/sb3_log/"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])


print(HumanoidEnv)


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

        print(return_vect)
        
        return return_vect

mod_env = ModifiedHumanoidEnv(list_of_obs=["position", "external_contact_forces"])

# '''

# env = gym.make("Humanoid-v5", render_mode="rgb_array")

model = PPO("MlpPolicy", mod_env, verbose=1)
model.set_logger(new_logger)

model.learn(total_timesteps=2e4)

# print(resp)

vec_env = model.get_env()
obs = vec_env.reset()

avg_reward = 0

for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    avg_reward += reward/1000
    # vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()

print(avg_reward)
# '''