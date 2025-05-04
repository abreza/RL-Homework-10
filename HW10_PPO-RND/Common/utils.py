import numpy as np
import torch
import random
import os
import gym
# import minigrid

def make_env(env_id, max_episode_steps=None):
    def _init():
        env = gym.make(env_id)
        if max_episode_steps:
            env._max_episode_steps = max_episode_steps
        return env
    return _init




def set_random_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_of_list(func):
    def wrapper(*args, **kwargs):
        lists = func(*args, **kwargs)
        return [
            sum(lst) / len(lst) for lst in lists[:-4]
        ] + [explained_variance(lists[-4], lists[-3])] + [explained_variance(lists[-2], lists[-1])]
    return wrapper


def explained_variance(ypred, y):
    """Compute 1 - Var[y - ypred] / Var[y]"""
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


class RunningMeanStd:
    """Tracks running mean and variance (like batch norm statistics)."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
