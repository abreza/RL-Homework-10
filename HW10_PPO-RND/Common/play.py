import gym
import numpy as np
import torch
import time
import os
from torch import device
from Common.utils import make_env
import gym_minigrid  # ensure gym_minigrid is installed

def extract_obs_image(obs):
    """Extracts the 'image' key from MiniGrid dict observation."""
    if isinstance(obs, tuple):
        obs, _ = obs
    if isinstance(obs, dict):
        return obs.get("image", obs)
    return obs

class Play:
    def __init__(self, env_name, agent, checkpoint, max_episode=1, render=True):
        # Make sure make_env() returns a gym environment instance, not a function
        self.env = make_env(env_name, 4500)()  # Add parentheses to call the function and create the env
        self.agent = agent
        self.agent.set_from_checkpoint(checkpoint)
        self.agent.set_to_eval_mode()
        self.device = device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_episode = max_episode
        self.render = render

    def evaluate(self):
        all_rewards = []

        for ep in range(self.max_episode):
            # Handling for different gym versions (with or without seed argument)
            try:
                raw_obs = self.env.reset(seed=ep)  # Gym >=0.26 supports seed argument
            except TypeError:
                raw_obs = self.env.reset()  # For older gym versions (no seed argument)

            obs = extract_obs_image(raw_obs)

            hidden_state = torch.zeros(1, 256).to(self.device)
            done = False
            episode_reward = 0

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action, *_ , hidden_state = self.agent.get_actions_and_values(obs_tensor, hidden_state)

                result = self.env.step(action)
                
                # Handle 4 or 5 values returned from env.step()
                if len(result) == 5:
                    next_obs_dict, ext_reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    next_obs_dict, ext_reward, done, info = result

                obs = extract_obs_image(next_obs_dict)

                episode_reward += ext_reward

                if self.render:
                    self.env.render()
                    time.sleep(0.05)

            print(f"Episode {ep + 1} Reward: {episode_reward:.2f}")
            all_rewards.append(episode_reward)

        print(f"\nAverage Reward over {self.max_episode} episodes: {np.mean(all_rewards):.2f}")
        self.env.close()

