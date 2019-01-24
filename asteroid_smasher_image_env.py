import gym
import arcade
import numpy as np
import asteroid_smasher_model
from gym import spaces


class AsteroidSmasherEnv(gym.Env):

    metadata = {'render.modes': ['human',]}

    valid_commands = [arcade.key.SPACE, arcade.key.LEFT, arcade.key.RIGHT, arcade.key.UP, arcade.key.DOWN]
    LIFE_SCORE_DECREMENT = 10

    def __init__(self):
        super(AsteroidSmasherEnv, self).__init__()
        N_CHANNELS = 1
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(len(self.valid_commands))
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(asteroid_smasher_model.SCREEN_HEIGHT, asteroid_smasher_model.SCREEN_WIDTH, N_CHANNELS), dtype=np.uint8)

        self.reset()

    def step(self, action):
        action = arcade.key.UP

        # perform action on environment
        if action not in self.previous_actions:
            for previous_action in self.previous_actions:
                self.window.on_key_release(previous_action, None)

        self.window.on_key_press(action, None)

        # update state
        time_since_last_update_call = 1.0
        self.window.update(time_since_last_update_call)

        # calculate reward
        reward = (self.previous_score - self.window.score) - (self.LIFE_SCORE_DECREMENT * (self.previous_lives - self.window.lives))

        # update saved state
        self.previous_score = self.window.score
        self.previous_lives = self.window.lives
        self.previous_actions = [action,]

        # create the observation image
        image = np.array(arcade.draw_commands.get_image().convert('LA'))
        obs = []
        for i in range(image.shape[0]):
            obs.append([])
            for j in range(image.shape[1]):
                obs[-1].append(min(image[i, j]))

        # return reward for action, episode has ended
        return np.array(obs), reward, self.window.game_over, {}


    def reset(self):
        self.window = asteroid_smasher_model.MyGame()
        self.window.start_new_game()
        self.previous_score = 0
        self.previous_lives = self.window.lives
        self.previous_actions = []


    def render(self, mode='human', close=False):
        if mode == 'human':
            pass
        elif mode == 'ai':
            pass
