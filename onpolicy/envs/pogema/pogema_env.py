import pogema.environment as environment
import pogema.grid_config as config
import gym


def PogemaENV(args):

    grid = config.Normal8x8(config.PredefinedDifficultyConfig())
    # env = environment._make_pogema(grid)
    envs = gym.make('Pogema-v0', config = grid)
    return envs
