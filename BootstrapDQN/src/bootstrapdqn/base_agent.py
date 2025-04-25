import time
from pathlib import Path
from typing import Union

import gymnasium as gym
import numpy as np
import torch

import wandb

from .replay_buffer import ReplayBuffer
from .wrappers import FFmpegVideoRecorder
from .utils import set_seed
import random


class BaseDQNAgent:
    def __init__(
        self,
        env_name,
        default_batch_size=64,
        replay_buffer_capacity=1000000,
        gamma=0.99,
        learning_rate=1e-4,
        tau=0.005,
        device="cuda" if torch.cuda.is_available() else "cpu",
        wandb_run=None,
        log_index="step",
        video_log_path_dir="videos",
        seed: int = 42,
        gradient_norm_clip: Union[float, None] = None,
        start_training_after: int = 1000,
    ):
        set_seed(seed)

        self.env = gym.make(env_name)

        self.video_log_path = Path(video_log_path_dir)
        self.video_log_path.mkdir(parents=True, exist_ok=True)
        self.eval_env = FFmpegVideoRecorder(
            gym.make(env_name, render_mode="rgb_array"), video_folder=str(self.video_log_path.resolve())
        )

        self.default_batch_size = default_batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.tau = tau
        self.device = device
        self.wandb_run = wandb_run
        self.log_index = log_index
        self.gradient_norm_clip = gradient_norm_clip
        self.start_training_after = start_training_after

        self._training_step = 0
        self._training_episode = 0
        self._cur_rollout_step = 0
        self._total_steps = 0

        self._actions = []

        self._create_replay_buffer(replay_buffer_capacity)

        self._create_network()
        self._hard_update_target_network()
        self._create_optimizer()

        self.train_mode()

    def eval_mode(self):
        """
        Set the agent to evaluation mode.
        """
        self.training = False
        self.q_network.eval()
        self.target_network.eval()
        self._loss = 0
        self._episode_loss = 0
        self._episode_reward = 0
        self._cur_rollout_step = 0
        self._actions.clear()

    def train_mode(self):
        """
        Set the agent to training mode.
        """
        self.training = True
        self.q_network.train()
        self._loss = 0
        self._episode_loss = 0
        self._episode_reward = 0
        self._cur_rollout_step = 0

    def add_experience(self, state, action, reward, next_state, done):
        """
        Add an experience to the replay buffer.
        """
        data = self._preprocess_add(state, action, reward, next_state, done)
        self.replay_buffer.add(**data)

    def act(self, state):
        """
        Take an action given the current state.

        Args:
            state (np.ndarray): The current state.
        """
        if self.training:
            self._training_step += 1
            action = self._act_in_training(state)
        else:
            action = self._act_in_eval(state)
        return action

    def _step(self, reward):
        """
        Step the agent with the given reward.
        """
        self._episode_reward += reward
        self._cur_rollout_step += 1
        self._total_steps += 1

    def _episode(self):
        """
        apply an episode
        """
        save_dict = self._wandb_train_episode_dict()
        if self.wandb_run is not None:
            if self.log_index == "step":
                self.wandb_run.log(save_dict, step=self._training_step)
            elif self.log_index == "episode":
                self.wandb_run.log(save_dict, step=self._training_episode)
            else:
                raise ValueError(f"Invalid log index {self.log_index}. Use 'step' or 'episode'.")

        self._episode_loss = 0
        self._episode_reward = 0
        self._cur_rollout_step = 0
        self._training_episode += 1

        self._actions.clear()

    def learn(self, batch_size=None):
        """
        Learn from the replay buffer.
        """
        self._loss = None
        if not self.training:
            print("Agent is not in training mode. It can not learn.")
            return

        if batch_size is None:
            batch_size = self.default_batch_size
        if len(self.replay_buffer) < batch_size or self._training_step < self.start_training_after:
            return

        batch = self.replay_buffer.sample(batch_size)
        loss = self._compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_norm_clip is not None:
            torch.nn.utils.clip_grad_norm_(self._learnable_parameters(), self.gradient_norm_clip)
        self.optimizer.step()
        self._soft_update_target_network()

        self._loss = loss.item()
        self._episode_loss += self._loss

        log_dict = self._wandb_train_step_dict()
        if self.wandb_run is not None:
            self.wandb_run.log(log_dict, step=self._training_step)

    def train(
        self,
        max_episodes=10000,
        max_steps_per_episode=5000,
        max_steps=1000000,
        max_time=4 * 60 * 60,
        learn_every=10,
        eval_every=10000,
    ):
        """
        Train the agent.
        """
        start_time = time.time()
        pre_evaluation_step = self._training_step
        max_steps += self._training_step
        finished = False

        for episode in range(max_episodes):
            self.train_mode()
            self.env.action_space.seed(random.randint(0, 1e32 - 1))
            state, _ = self.env.reset(seed=random.randint(0, 1e32 - 1))
            for step in range(max_steps_per_episode):
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self._step(reward)
                self.add_experience(state, action, reward, next_state, done)

                state = next_state

                if self._total_steps % learn_every == 0:
                    self.learn()

                if done:
                    break

                if self._training_step >= max_steps or time.time() - start_time >= max_time:
                    finished = True
                    break

            self._episode()

            if self._training_step - pre_evaluation_step >= eval_every:
                self.evaluate()
                pre_evaluation_step = self._training_step

            if finished:
                self.evaluate()
                print(f"Trained for {self._training_step} steps.")
                break

    def save(self, path, save_replay_buffer=True):
        """
        Save the agent's state.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if save_replay_buffer:
            self.replay_buffer.save(path)

        save_dict = self._save_dict()
        torch.save(save_dict, path / "agent.pth")

    @classmethod
    def load(cls, path, device="cuda" if torch.cuda.is_available() else "cpu", use_replay_buffer=True):
        """
        Load the agent's state.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")
        if path.is_dir():
            if not (path / "agent.pth").exists():
                raise FileNotFoundError(f"Agent file not found in {path}. there should be an `agent.pth` file.")
        else:
            raise ValueError(f"Path {path} is not a directory.")

        checkpoint = torch.load(path / "agent.pth", map_location=device, weights_only=False)
        agent = cls(env_name=checkpoint["env_name"], device=device)

        print("Loading optimizer.")
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        checkpoint.pop("optimizer")

        for key in checkpoint["networks"]:
            if key in vars(agent):
                print(f"Loading {key} from checkpoint.")
                getattr(agent, key).load_state_dict(checkpoint["networks"][key])
            else:
                print(f"Warning: {key} not found in checkpoint. Using default value.")

        if "wandb_run" in checkpoint:
            print("Loading wandb run.")
            agent.wandb_run = wandb.init(
                project=checkpoint["wandb_project"],
                config=checkpoint["wandb_config"],
                id=checkpoint["wandb_run"],
                resume="must",
            )
            checkpoint.pop("wandb_run")

        if "rng_state" in checkpoint:
            print("Loading RNG state.")
            torch.set_rng_state(checkpoint["rng_state"]["torch"].cpu())
            np.random.set_state(checkpoint["rng_state"]["numpy"])
            random.setstate(checkpoint["rng_state"]["random"])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(
                    [checkpoint["rng_state"]["torch_cuda"][i].cpu() for i in range(torch.cuda.device_count())]
                )
            agent.env.action_space.seed(checkpoint["rng_state"]["gym"])
            agent.eval_env.action_space.seed(checkpoint["rng_state"]["gym_eval"])
            checkpoint.pop("rng_state")

        save_dict = agent._save_dict()
        for key in checkpoint:
            if key in save_dict:
                print(f"Loading {key} from checkpoint.")
                setattr(agent, key, checkpoint[key])
            else:
                print(f"Warning: {key} not found in checkpoint. Using default value.")

        if use_replay_buffer:
            if (path / "replay_buffer.pth").exists():
                agent.replay_buffer.load(path)
            else:
                raise FileNotFoundError(
                    f"Replay buffer not found in {path}. there should be a `replay_buffer.pth` file."
                )

        return agent

    def evaluate(self, video_name="video", max_steps=5000):
        self.eval_mode()
        self.eval_env.action_space.seed(random.randint(0, 1e32 - 1))
        state, _ = self.eval_env.reset(video_name=video_name, seed=random.randint(0, 1e32 - 1))
        for step in range(max_steps):
            action = self.act(state)
            self._actions.append(action)
            next_state, reward, terminated, truncated, _ = self.eval_env.step(action)
            done = terminated or truncated
            self._step(reward)
            state = next_state

            if done:
                break

        if self.wandb_run is not None:
            eval_dict = self._wandb_eval_dict()
            if self.log_index == "step":
                self.wandb_run.log(eval_dict, step=self._training_step)
            elif self.log_index == "episode":
                self.wandb_run.log(eval_dict, step=self._training_episode)
            else:
                raise ValueError(f"Invalid log index {self.log_index}. Use 'step' or 'episode'.")

    def _create_replay_buffer(self, max_size=1000000):
        """
        Create a replay buffer for storing experiences.
        """
        self.replay_buffer = ...
        raise NotImplementedError(
            "Replay buffer should be initialized in child classes. Use `ReplayBuffer` class to create a replay buffer."
        )

    def _create_network(self):
        """
        Create the neural network for the agent.
        """
        self.q_network = ...
        self.target_network = ...
        raise NotImplementedError(
            "Network should be initialized in child classes. Don't forget to set the target network to eval mode."
        )

    def _learnable_parameters(self):
        """
        Return the learnable parameters of the agent.
        """
        return self.q_network.parameters()

    def _create_optimizer(self):
        """
        Create the optimizer for the agent.
        """
        self.optimizer = torch.optim.Adam(self._learnable_parameters(), lr=self.learning_rate)

    def _soft_update_target_network(self):
        """
        Soft update the target network.
        """
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def _hard_update_target_network(self):
        """
        Hard update the target network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _compute_loss(self, batch: dict):
        """
        Compute the loss for the given batch.

        Args:
            batch (dict): A batch of experiences.
        """
        raise NotImplementedError("Loss computation should be implemented in child classes.")

    def _preprocess_add(self, state, action, reward, next_state, done) -> dict:
        """
        Preprocess the experience before adding it to the replay buffer.

        Returns:
            dict: A dictionary containing the preprocessed experience.
        """

        reward = self._reward_transformation(reward)

        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)
        return {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }

    def _act_in_training(self, state):
        """
        Take an action in training mode.
        """
        raise NotImplementedError("Action selection in training mode should be implemented in child classes.")

    def _act_in_eval(self, state):
        """
        Take an action in evaluation mode.
        """
        raise NotImplementedError("Action selection in evaluation mode should be implemented in child classes.")

    def _wandb_train_step_dict(self):
        """
        Create a dictionary for logging the training step.
        """
        return {"train_step/loss": self._loss}

    def _wandb_train_episode_dict(self):
        """
        Create a dictionary for logging the training episode.
        """
        return {
            "train_episode/sum_loss": self._episode_loss,
            "train_episode/sum_reward": self._episode_reward,
            "train_episode/episode_length": self._cur_rollout_step,
            "train_episode/mean_reward": self._episode_reward / self._cur_rollout_step,
            "train_episode/mean_loss": self._episode_loss / self._cur_rollout_step,
        }

    def _wandb_eval_dict(self):
        """
        Create a dictionary for logging the evaluation.
        """
        return {
            "eval_episode/sum_reward": self._episode_reward,
            "eval_episode/episode_length": self._cur_rollout_step,
            "eval_episode/mean_reward": self._episode_reward / self._cur_rollout_step,
            "eval_episode/action_histogram": wandb.Histogram(self._actions),
            "eval_video/video": wandb.Video(self.eval_env.get_path(), format="mp4"),
        }

    def _save_dict(self):
        """
        Dictionary of the agent's state.
        """
        save_dict = {
            "training": self.training,
            "gamma": self.gamma,
            "tau": self.tau,
            "learning_rate": self.learning_rate,
            "default_batch_size": self.default_batch_size,
            "env_name": self.env.spec.id,
            "_training_step": self._training_step,
            "_training_episode": self._training_episode,
            "_cur_rollout_step": self._cur_rollout_step,
            "_total_steps": self._total_steps,
            "optimizer": self.optimizer.state_dict(),
        }

        save_dict["networks"] = {}
        for key in vars(self):
            if isinstance(getattr(self, key), torch.nn.Module):
                save_dict["networks"][key] = getattr(self, key).state_dict()

        if self.wandb_run is not None:
            save_dict["wandb_run"] = self.wandb_run.id
            save_dict["wandb_config"] = dict(self.wandb_run.config)
            save_dict["wandb_project"] = self.wandb_run.project

        save_dict["rng_state"] = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "gym": self.env.action_space.seed(),
            "gym_eval": self.eval_env.action_space.seed(),
        }
        return save_dict

    def _reward_transformation(self, reward):
        """
        Transform the reward to a suitable range.
        """
        return reward
