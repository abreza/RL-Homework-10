from torch.multiprocessing import Process
import numpy as np
from Common.utils import make_env

class Worker(Process):
    def __init__(self, id, conn, **config):
        super(Worker, self).__init__()
        self.id = id
        self.conn = conn
        self.config = config
        self.env_name = config["env_name"]
        self.max_episode_steps = config["max_frames_per_episode"]
        self.render_mode = config.get("render", False)
        self.env = make_env(self.env_name, self.max_episode_steps)()
        self.reset()

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):  # Gym >= 0.26
            obs, _ = obs
        self.obs = obs
        self.steps = 0

    def render(self):
        self.env.render()

    def run(self):
        while True:
            self.conn.send(self.obs)
            action = self.conn.recv()

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.steps += 1

            if self.steps >= self.max_episode_steps:
                done = True

            if self.render_mode:
                self.render()

            if isinstance(next_obs, tuple):  # Gym >= 0.26
                next_obs, _ = next_obs

            self.conn.send((next_obs, reward, done, info))

            if done:
                self.reset()
            else:
                self.obs = next_obs
