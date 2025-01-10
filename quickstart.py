import gymnasium as gym

from stable_baselines3.common.logger import configure

from stable_baselines3 import PPO
import numpy as np

from src.modified_env import ModifiedHumanoidEnv


tmp_path = "/tmp/sb3_log/"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])


# print(HumanoidEnv)




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