import gym
import arcade
import numpy as np
import asteroid_smasher
from gym import spaces
from pyautogui import keyDown, keyUp

OFFSCREEN_SPACE = 10
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

class AsteroidSmasherBaseEnv(gym.Env):
    pass

class AsteroidSmasherTrainEnv(gym.Env):

    metadata = {'render.modes': ['human',]}

    movement_commands = [[arcade.key.UP,],
                         [arcade.key.UP, arcade.key.LEFT],
                         [arcade.key.UP, arcade.key.RIGHT],
                         [arcade.key.DOWN,],
                         [arcade.key.DOWN, arcade.key.LEFT],
                         [arcade.key.DOWN, arcade.key.RIGHT],
                         [arcade.key.LEFT,],
                         [arcade.key.RIGHT,],]
    valid_commands = [[arcade.key.SPACE,],] + movement_commands + [movement + [arcade.key.SPACE] for movement in movement_commands]

    LIFE_SCORE_DECREMENT = 10
    BIN_PIXEL_SIZE = 20
    BULLET_VALUE = 5
    PLAYER_VALUE = 6

    def __init__(self, verbose=False):
        super(AsteroidSmasherTrainEnv, self).__init__()
        self.window = None
        self.verbose=verbose
        N_CHANNELS = 1
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(len(self.valid_commands))
        self.observation_space = spaces.Box(low=0, high=6, shape=(1273,))

        self.window = asteroid_smasher.MyGame(verbose=verbose)
        self.reset()

    def step(self, actions):
        actions = self.valid_commands[actions]

        if self.verbose:
            print("Step: " + str(actions))

        # perform action on environment
        for action in actions:
            if action not in self.previous_actions:
                self.window.on_key_press(action, None)

        for previous_action in self.previous_actions:
            if previous_action not in actions:
                self.window.on_key_release(previous_action, None)

        # update state, we skip 15 update frames
        for _ in range(15):
            time_since_last_update_call = 1.0
            self.window.update(time_since_last_update_call)

        # calculate reward
        reward = (self.window.score - self.previous_score) - (self.LIFE_SCORE_DECREMENT * (self.previous_lives - self.window.lives))
        if self.verbose:
            print("Reward: " + str(reward))

        # update saved state
        self.previous_score = self.window.score
        self.previous_lives = self.window.lives
        self.previous_actions = actions
        if reward != 0:
            print(reward)
        # return reward for action, episode has ended
        return self.get_state(), reward, self.window.game_over, {}


    def reset(self):
        self.window.start_new_game()
        self.previous_score = 0
        self.previous_lives = self.window.lives
        self.previous_actions = []
        return self.get_state()

    def get_state(self):
        if self.verbose:
            print("create the current state")

        obs = []

        # add player features
        player_sprite = self.window.player_sprite
        obs.append(int(player_sprite.respawning > 0))
        obs.append(player_sprite.angle)

        asset_array = np.zeros(shape=(int((SCREEN_WIDTH + 2 * OFFSCREEN_SPACE)/self.BIN_PIXEL_SIZE),
                                      int((SCREEN_HEIGHT + 2 * OFFSCREEN_SPACE)/self.BIN_PIXEL_SIZE)),
                               dtype=int)

        if self.verbose:
            print("set bullets")
        for bullet in self.window.bullet_list:
            asset_array[int(bullet.center_x/self.BIN_PIXEL_SIZE), int(bullet.center_y/self.BIN_PIXEL_SIZE)] = self.BULLET_VALUE

        if self.verbose:
            print("set asteroids")
        for asteroid in self.window.asteroid_list:
            asset_array[int(asteroid.center_x/self.BIN_PIXEL_SIZE), int(asteroid.center_y/self.BIN_PIXEL_SIZE)] = asteroid.size

        if self.verbose:
            print("set player")
        asset_array[int(player_sprite.center_x/self.BIN_PIXEL_SIZE), int(player_sprite.center_y/self.BIN_PIXEL_SIZE)] = self.PLAYER_VALUE

        return np.array(obs + list(asset_array.flatten()))


    def render(self, mode='human', close=False):
        if mode == 'human':
            pass
        elif mode == 'ai':
            pass


class AsteroidSmasherTestEnv(gym.Env):

    metadata = {'render.modes': ['human',]}

    movement_commands = [[arcade.key.UP,],
                         [arcade.key.UP, arcade.key.LEFT],
                         [arcade.key.UP, arcade.key.RIGHT],
                         [arcade.key.DOWN,],
                         [arcade.key.DOWN, arcade.key.LEFT],
                         [arcade.key.DOWN, arcade.key.RIGHT],
                         [arcade.key.LEFT,],
                         [arcade.key.RIGHT,],]
    valid_commands = [[arcade.key.SPACE,],] + movement_commands + [movement + [arcade.key.SPACE] for movement in movement_commands]

    KEY_MAPPING = {arcade.key.UP: 'up',
                   arcade.key.LEFT: 'left',
                   arcade.key.RIGHT: 'right',
                   arcade.key.LEFT: 'left',
                   arcade.key.SPACE: 'space'}

    LIFE_SCORE_DECREMENT = 10
    BIN_PIXEL_SIZE = 20
    BULLET_VALUE = 5
    PLAYER_VALUE = 6

    def __init__(self, verbose=False):
        super(AsteroidSmasherTestEnv, self).__init__()
        self.window = None
        self.verbose=verbose
        N_CHANNELS = 1
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(len(self.valid_commands))
        self.observation_space = spaces.Box(low=0, high=6, shape=(1273,))

        self.window = asteroid_smasher.MyGame(verbose=verbose)


    def step(self, actions):
        actions = self.valid_commands[actions]

        if self.verbose:
            print("Step: " + str(actions))

        # perform action on environment
        for action in actions :
            if action not in self.previous_actions:
                if self.verbose:
                    print("Press Key: " + str(action))
                keyDown(self.KEY_MAPPING[action])

        for previous_action in self.previous_actions:
            if previous_action not in actions:
                if self.verbose:
                    print("Release Key: " + str(previous_action))
                keyUp(self.KEY_MAPPING[action])

        if self.verbose:
            print("update saved state")
        self.previous_actions = actions

        return None, None, None, None


    def reset(self):
        self.window.start_new_game()
        self.previous_score = 0
        self.previous_lives = self.window.lives
        self.previous_actions = []
        arcade.run()
        return self.get_state()

    def get_state(self):
        if self.verbose:
            print("create the current state")
        obs = []

        # add player features
        player_sprite = self.window.player_sprite
        obs.append(int(player_sprite.respawning > 0))
        obs.append(player_sprite.angle)

        asset_array = np.zeros(shape=(int((SCREEN_WIDTH + 2 * OFFSCREEN_SPACE)/self.BIN_PIXEL_SIZE),
                                      int((SCREEN_HEIGHT + 2 * OFFSCREEN_SPACE)/self.BIN_PIXEL_SIZE)),
                               dtype=int)

        if self.verbose:
            print("set bullets")
        for bullet in self.window.bullet_list:
            asset_array[int(bullet.center_x/self.BIN_PIXEL_SIZE), int(bullet.center_y/self.BIN_PIXEL_SIZE)] = self.BULLET_VALUE

        if self.verbose:
            print("set asteroids")
        for asteroid in self.window.asteroid_list:
            asset_array[int(asteroid.center_x/self.BIN_PIXEL_SIZE), int(asteroid.center_y/self.BIN_PIXEL_SIZE)] = asteroid.size

        if self.verbose:
            print("set player")
        asset_array[int(player_sprite.center_x/self.BIN_PIXEL_SIZE), int(player_sprite.center_y/self.BIN_PIXEL_SIZE)] = self.PLAYER_VALUE

        return np.array(obs + list(asset_array.flatten()))


    def render(self, mode='human', close=False):
        if mode == 'human':
            pass
        elif mode == 'ai':
            pass