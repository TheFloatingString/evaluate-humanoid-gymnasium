import gymnasium as gym

from stable_baselines3.common.logger import configure

from stable_baselines3 import PPO
import numpy as np

from src.modified_env import ModifiedHumanoidEnv


def run_experiment(is_position: bool, is_velocity: bool, is_com_inertia: bool, is_com_velocity: bool, is_actuator_forces: bool, is_external_contact_forces: bool):
    return True


for is_position in [True, False]:
    for is_velocity in [True, False]:
        for is_com_inertia in [True, False]:
            for is_com_velocity in [True, False]:
                for is_actuator_forces in [True, False]:
                    for is_external_contact_forces in [True, False]:
                        run_experiment(is_position=is_position, is_velocity=is_velocity, is_com_inertia=is_com_inertia, is_com_velocity=is_com_velocity,  is_actuator_forces=is_actuator_forces, is_external_contact_forces=is_external_contact_forces)
                        



raise KeyError

tmp_path = "/tmp/sb3_log/"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

mod_env = ModifiedHumanoidEnv(list_of_obs=["position", "external_contact_forces"])

model = PPO("MlpPolicy", mod_env, verbose=1)
model.set_logger(new_logger)
model.learn(total_timesteps=2e4)

# vec_env = model.get_env()
# obs = vec_env.reset()

# avg_reward = 0

# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     avg_reward += reward/1000

# print(avg_reward)
