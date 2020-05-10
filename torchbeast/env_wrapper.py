# The MIT License
#
# Copyright (c) 2017 OpenAI (http://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in # all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Taken from
#   https://raw.githubusercontent.com/openai/baselines/7c520852d9cf4eaaad326a3d548efc915dc60c10/baselines/common/atari_wrappers.py
# and slightly modified by (c) Facebook, Inc. and its affiliates.
# 2 May 2020 Modified by urw7rs

import torch
import numpy as np
import gym
import cv2

cv2.ocl.setUseOpenCL(False)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=64, height=64):
        super().__init__(env)
        self._width = width
        self._height = height

        original_space = env.observation_space
        c = original_space.shape[-1]
        new_space = gym.spaces.Box(
            low=0, high=255, shape=(self._height, self._width, c), dtype=np.uint8,
        )
        self.observation_space = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

        if c == 1:
            self._grayscale = True
        else:
            self._grayscale = False

    def observation(self, obs):
        frame = obs

        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        return frame


class ToTensor(gym.ActionWrapper):
    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)

    def step(self, action):
        return self.env.step(self.action(action))

    def action(self, action):
        action = dict(zip(self.env.order, action.astype(np.int64).tolist()))
        return action

    def reverse_action(self, action):
        return np.asarray([action.values()], dtype=np.int64)


class AddDim(gym.Wrapper):
    """
    add T and B dimension to observation
    """

    def step(self, action):
        action = action.view(action.shape[-1]).numpy()
        obs, reward, done, info = self.env.step(action)
        obs = torch.as_tensor(obs).view((1, 1) + obs.shape)
        reward = torch.as_tensor(reward).view(1, 1)
        done = torch.as_tensor(done).view(1, 1)

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = torch.as_tensor(obs).view((1, 1) + obs.shape)
        return obs


def make_raw(env_id, config):
    env = gym.make("spiral:" + env_id)
    if len(config) > 0:
        env.configure(**config)
    return ToTensor(env)


def wrap_deepmind(env, **kwargs):
    return WarpFrame(env, **kwargs)


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))


def wrap_pytorch(env):
    return ImageToPyTorch(env)
