import asteroid_smasher_eng_env
import gym
from apscheduler.schedulers.background import BackgroundScheduler

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, PPO2
import threading

def main():
    asteroid = True
    train = False

    if asteroid:
        if train:
            env = asteroid_smasher_eng_env.AsteroidSmasherTrainEnv()
        else:
            env = asteroid_smasher_eng_env.AsteroidSmasherTestEnv(verbose=True)
    else:
        env = gym.make('MsPacman-ram-v0')
    env = DummyVecEnv([lambda: env])

    if train:
        model = PPO2(MlpPolicy, env, ent_coef=0.1, verbose=1)
        # Train the agent
        model.learn(total_timesteps=1000000)
        # Save the agent
        if asteroid:
            model.save("asteroid_smasher_model/ppo2_model")
        else:
            model.save("pacman/ppo2_model")
        del model  # delete trained model to demonstrate loading

        return

    # Load the trained agent
    if asteroid:
        model = PPO2.load("asteroid_smasher_model/ppo2_model", env=env)
    else:
        model = PPO2.load("pacman/ppo2_model", env=env)

    def play_game():
        print("step")
        action, _states = model.predict(env.envs[0].get_state())
        obs, rewards, dones, info = env.step([action,])

    scheduler = BackgroundScheduler()
    scheduler.add_job(play_game, 'interval', seconds = 0.25)
    scheduler.start()

    # Play trained agent
    obs = env.reset()

if __name__ == "__main__":
    main()