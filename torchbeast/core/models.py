# Copyright urw7rs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

import nest

from collections import OrderedDict


class Net(nn.Module):
    def __init__(self, obs_shape, action_shape, grid_shape, order):
        super(Net, self).__init__()
        self._action_shape = action_shape
        self._order = order
        self._num_actions = len(order)

        c, h, w = obs_shape
        self.register_buffer("grid", self._grid(h, w))

        self.conv5x5 = nn.Conv2d(c + 2, 32, 5, 1, 2)

        self.action_fc = nn.Sequential(
            ActionMask(self._order),
            MLP(action_shape, grid_shape),
            Linear(16 * len(action_shape), 64, 32),
            View(-1, 32, 1, 1),
        )

        self.fc = Linear(10, 64, 32)
        self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(),
        )

        self.resblock = nn.Sequential(*[ResBlock(32) for _ in range(8)])

        self.flatten_fc = nn.Sequential(nn.Flatten(1, 3), nn.Linear(8 * 8 * 32, 256))

        self.lstm = nn.LSTM(256, 256, num_layers=1)
        self.policy = Decoder(order, action_shape, grid_shape)
        self.baseline = nn.Linear(256, 1)

    def _grid(self, w, h):
        x_grid = torch.linspace(-1, 1, w)
        x_grid = x_grid.view(1, 1, 1, w)
        x_grid = x_grid.repeat(1, 1, w, 1)

        y_grid = torch.linspace(-1, 1, h)
        y_grid = y_grid.view(1, 1, h, 1)
        y_grid = y_grid.repeat(1, 1, 1, h)

        return torch.cat([y_grid, x_grid], dim=1)

    def initial_action(self, batch_size=1):
        return torch.zeros(1, batch_size, self._num_actions).long()

    def initial_state(self, batch_size=1):
        return tuple(torch.zeros(1, batch_size, 256) for _ in range(2))

    def forward(self, input, core_state):
        T, B, *_ = input["obs"].shape
        grid = self.grid.repeat(T * B, 1, 1, 1)

        notdone = (~input["done"]).float()
        action = torch.flatten(input["action"] * notdone.unsqueeze(dim=2), 0, 1)
        obs = torch.flatten(input["obs"].float(), 0, 1)
        noise = torch.flatten(input["noise"], 0, 1)

        spatial = self.conv5x5(torch.cat([obs, grid], dim=1))
        noise_embedding = self.fc(noise).view(-1, 32, 1, 1)
        mlp = self.action_fc(action)

        embedding = self.relu(spatial + noise_embedding + mlp)

        h = self.conv(embedding)
        h = self.resblock(h)
        h = self.flatten_fc(h)
        h = self.relu(h)

        core_input = h.view(T, B, 256)
        core_output_list = []
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            nd = nd.view(1, -1, 1)
            core_state = nest.map(nd.mul, core_state)
            output, core_state = self.lstm(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)

        action, policy_logits = self.policy(core_output, action)
        baseline = self.baseline(core_output)

        action = action.view(T, B, self._num_actions)
        baseline = baseline.view(T, B)
        policy_logits = nest.map(lambda t: t.view(T, B, -1), policy_logits)

        return (action, policy_logits, baseline), core_state


class Decoder(nn.Module):
    SPATIAL_ACTIONS = ["end", "control"]

    def __init__(self, order, action_shape, grid_shape):
        super(Decoder, self).__init__()
        self.num_actions = len(order)
        modules = []
        for i in range(self.num_actions):
            if i < 2:
                module = [View(-1, 16, 4, 4)]
                module.append(nn.ConvTranspose2d(16, 32, 4, 2, 1))
                module.extend([ResBlock(32) for i in range(8)])
                module.extend([nn.ConvTranspose2d(32, 32, 4, 2, 1) for i in range(2)])
                module.extend([nn.Conv2d(32, 1, 3, 1, 1), View(-1, 32 * 32)])
                module = nn.Sequential(*module)
            else:
                module = nn.Linear(256, action_shape[i])
            modules.append(module)
        self.decode = nn.ModuleList(modules)

        self.mlp = MLP(action_shape, grid_shape)
        self.concat_fc = nn.Sequential(nn.Linear(16 + 256, 256), nn.ReLU(inplace=True))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, h, actions=None):
        logits = []
        if self.training:
            for i in range(self.num_actions):
                logit = self.decode[i](h)

                concat = torch.cat(
                    [h, self.mlp[i](actions[:, i : i + 1].float())], dim=1
                )
                residual = self.concat_fc(concat)
                h = self.relu(h + residual)

                logits.append(logit)

            return actions, logits
        else:
            actions = []
            for i in range(self.num_actions):
                logit = self.decode[i](h)
                action = torch.multinomial(F.softmax(logit, dim=1), num_samples=1)

                concat = torch.cat([h, self.mlp[i](action.float())], dim=1)
                residual = self.concat_fc(concat)
                h = self.relu(h + residual)

                actions.append(action)
                logits.append(logit)

            actions = torch.cat(actions, dim=1)
            return actions, logits


class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()
        modules = [nn.ConvTranspose2d(16, 32, 4, 2, 1)]
        for i in range(8):
            modules.append(ResBlock(32))
        for i in range(2):
            modules.append(nn.ConvTranspose2d(32, 32, 4, 2, 1))
        modules.append(nn.Conv2d(32, 1, 3, 1, 1))
        self.conv = nn.Sequential(*modules)

    def forward(self, z):
        output = self.conv(z.view(-1, 16, 4, 4))
        return output.view(-1, 32 * 32)


class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.size = args

    def forward(self, x):
        return x.view(*self.size)


class ActionMask(nn.Module):
    def __init__(self, order):
        super(ActionMask, self).__init__()
        move_mask = OrderedDict([(key, 0.0) for key in order])

        for k in ["control", "end", "flag"]:
            if k in order:
                move_mask[k] = 1.0

        move = torch.tensor([list(move_mask.values())])
        paint = torch.ones(1, len(order))
        self.register_buffer("move", move)
        self.register_buffer("paint_minus_move", paint - move)
        self.i = 2  # flag index

    def forward(self, action):
        flag = action[:, self.i : self.i + 1]
        # equal to mask = (1.0 - flag) * self.move + flag * self.paint
        mask = self.move + flag * self.paint_minus_move

        return action * mask


class MLP(nn.Module):
    def __init__(self, action_shape, grid_shape):
        super(MLP, self).__init__()
        self.w, self.h = grid_shape
        modules = []
        for i, shape in enumerate(action_shape):
            if i < 2:
                module = Location(grid_shape)
            else:
                module = Scalar(shape)
            modules.append(module)

        self.mlp_list = nn.ModuleList(modules)

    def __getitem__(self, idx):
        return self.mlp_list[idx]

    def forward(self, action):
        actions = action.unbind(dim=1)
        y = [
            self.mlp_list[i](action.unsqueeze(dim=1))
            for i, action in enumerate(actions)
        ]
        return torch.cat(y, dim=1)


class Location(nn.Module):
    def __init__(self, shape):
        super(Location, self).__init__()
        self.w, self.h = shape
        self.linear = nn.Linear(2, 16)

    def forward(self, action):
        remainder = torch.fmod(action, self.w)
        x = -1.0 + 2.0 * (action - remainder) / (self.w - 1.0)
        y = -1.0 + 2.0 * remainder / (self.h - 1.0)
        action = torch.stack([y, x]).view(-1, 2)
        return self.linear(action)


class Scalar(nn.Module):
    def __init__(self, shape):
        super(Scalar, self).__init__()
        self.linear = nn.Linear(shape, 16)
        self._shape = shape

    def forward(self, action):
        action = F.one_hot(action.squeeze(dim=1).long(), self._shape).float()
        return self.linear(action)


class Linear(nn.Module):
    def __init__(self, *features):
        super(Linear, self).__init__()
        modules = []
        in_features = features[0]
        for out_features in features[1:]:
            modules.extend(
                [nn.Linear(in_features, out_features), nn.ReLU(inplace=True)]
            )
            in_features = out_features
        self.linear = nn.Sequential(*modules)

    def forward(self, x):
        return self.linear(x)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.conv(x)
        residual += x
        return self.relu(residual)


class Condition(nn.Module):
    def __init__(self, model):
        super(Condition, self).__init__()
        self.model = model

    def initial_action(self, batch_size=1):
        return self.model.initial_action(batch_size)

    def initial_state(self, batch_size=1):
        return self.model.initial_state(batch_size)

    def forward(self, input, core_state):
        obs = input["obs"]
        condition = input.pop("condition")
        T, *_ = input["obs"].shape
        input["obs"] = torch.cat([obs, condition.repeat(T, 1, 1, 1, 1)], dim=2)
        return self.model(input, core_state)


class Discriminator(nn.Module):
    def __init__(self, obs_shape, power_iters):
        super(Discriminator, self).__init__()
        c, h, w = obs_shape

        ndf = 64
        self.main = nn.Sequential(
            # (c) x 64 x 64
            nn.Conv2d(c, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            # (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            # (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            # (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
            # (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

        for module in self.main.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
                nn.utils.spectral_norm(module, n_power_iterations=power_iters)

    def forward(self, obs):
        x = self.main(obs)
        if self.training:
            return x
        else:
            return torch.sigmoid(x)


class Conditional(nn.Module):
    def __init__(self, D):
        super(Conditional, self).__init__()
        self.D = D

    def forward(self, obs, condition):
        half = obs.shape[-1] // 2
        condition[:, :, :, half:] = obs[:, :, :, half:]
        cat = torch.cat([obs, condition], dim=1)
        return self.D(cat)
