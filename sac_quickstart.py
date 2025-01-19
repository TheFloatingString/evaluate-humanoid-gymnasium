import gymnasium as gym

from stable_baselines3.common.logger import configure

from stable_baselines3 import SAC
import numpy as np

from src.modified_env import ModifiedHumanoidEnv
import tqdm

### SELECT PARAMETERS
TOTAL_TIMESTEPS = 2e4


def generate_log_folderpath(is_position: bool, is_velocity: bool, is_com_inertia: bool, is_com_velocity: bool, is_actuator_forces: bool, is_external_contact_forces: bool) -> str:
    folderpath = "tmp/v3/sac/sb3_sac_humanoid-"
    if is_position:
        folderpath += "-position"
    if is_velocity:
        folderpath += "-velocity"
    if is_com_inertia:
        folderpath += "-com_inertia"
    if is_com_velocity:
        folderpath += "-com_velocity"
    if is_actuator_forces:
        folderpath += "-actuator_forces"
    if is_external_contact_forces:
        folderpath += "-external_contact_forces"
    folderpath += '/'
    return folderpath
    

def run_experiment(is_position: bool, is_velocity: bool, is_com_inertia: bool, is_com_velocity: bool, is_actuator_forces: bool, is_external_contact_forces: bool):

    log_folderpath = generate_log_folderpath(is_position=is_position, is_velocity=is_velocity, is_com_inertia=is_com_inertia, is_com_velocity=is_com_velocity,  is_actuator_forces=is_actuator_forces, is_external_contact_forces=is_external_contact_forces)

    new_logger = configure(log_folderpath, ["stdout", "csv", "tensorboard"])

    list_of_obs = list()
    if is_position:
        list_of_obs.append("position")
    if is_velocity: 
        list_of_obs.append("velocity")
    if is_com_inertia:
        list_of_obs.append("com_inertia")
    if is_com_velocity:
        list_of_obs.append("com_velocity")
    if is_actuator_forces:
        list_of_obs.append("actuator_forces")
    if is_external_contact_forces:
        list_of_obs.append("external_contact_forces")

    mod_env = ModifiedHumanoidEnv(list_of_obs=list_of_obs)

    model = SAC("MlpPolicy", mod_env, verbose=0)
    model.set_logger(new_logger)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    return True

counter = 0
total_exp = 2**6

N_TOTAL = 2**6

with tqdm.tqdm(total=N_TOTAL) as p_bar:
    for is_position in [True, False]:
        for is_velocity in [True, False]:
            for is_com_inertia in [True, False]:
                for is_com_velocity in [True, False]:
                    for is_actuator_forces in [True, False]:
                        for is_external_contact_forces in [True, False]:
                            print(f"Experiment run: {counter}/{total_exp}")
                            run_experiment(is_position=is_position, is_velocity=is_velocity, is_com_inertia=is_com_inertia, is_com_velocity=is_com_velocity,  is_actuator_forces=is_actuator_forces, is_external_contact_forces=is_external_contact_forces)
                            # counter += 1
                            p_bar.update(1)
pbar.close()
