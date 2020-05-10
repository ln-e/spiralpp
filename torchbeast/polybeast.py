# Copyright (c) Facebook, Inc. and its affiliates.
# 2 May 2020 - Modified by urw7rs
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

import argparse
import collections
import logging
import os
import signal
import subprocess
import threading
import time
import timeit
import traceback

os.environ["OMP_NUM_THREADS"] = "1"  # noqa Necessary for multithreading.

import nest
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from queue import Queue
from libtorchbeast import actorpool
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CelebA, Omniglot, MNIST

from torch.utils.data import DataLoader
from torchbeast.core import file_writer
from torchbeast.core import vtrace
from torchbeast.core import models
from torchbeast.core import datasets

from torchbeast import env_wrapper


# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

parser.add_argument("--pipes_basename", default="unix:/tmp/polybeast",
                    help="Basename for the pipes for inter-process communication. "
                    "Has to be of the type unix:/some/path.")
parser.add_argument("--mode", default="train",
                    choices=["train", "test", "test_render"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")
parser.add_argument("--start_servers", dest="start_servers", action="store_true",
                    help="Spawn polybeast_env servers automatically.")
parser.add_argument("--no_start_servers", dest="start_servers", action="store_false",
                    help="Don't spawn polybeast_env servers automatically.")
parser.set_defaults(start_servers=True)

# Environment settings
parser.add_argument("--env_type", type=str, default="libmypaint",
                    help="Environment. Ignored if --no_start_servers is passed.")
parser.add_argument("--episode_length", type=int, default=20,
                    help="Set epiosde length")
parser.add_argument("--canvas_width", type=int, default=256,
                    help="Set canvas render width")
parser.add_argument("--brush_type", type=str, default="classic/dry_brush",
                    help="Set brush type from brush dir")
parser.add_argument("--brush_sizes", nargs='+', type=int,
                    default=[1, 2, 4, 8, 12, 24],
                    help="Set brush_sizes float is allowed")
parser.add_argument("--use_pressure", action="store_true",
                    help="use_pressure flag")
parser.add_argument("--use_compound", action="store_true",
                    help="use compound action space")
parser.add_argument("--new_stroke_penalty", type=float, default=0.0,
                    help="penalty for new stroke")
parser.add_argument("--stroke_length_penalty", type=float, default=0.0,
                    help="penalty for stroke length")

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--savedir", default="~/logs/torchbeast",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=4, type=int, metavar="N",
                    help="Number of actors.")
parser.add_argument("--total_steps", default=100000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--batch_size", default=64, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--num_learner_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--num_inference_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")
parser.add_argument("--max_learner_queue_size", default=None, type=int, metavar="N",
                    help="Optional maximum learner queue size. Defaults to batch_size.")
parser.add_argument("--unroll_length", default=20, type=int, metavar="T",
                    help="The unroll length (time dimension).")
parser.add_argument("--condition", action="store_true",
                    help='condition flag')
parser.add_argument("--use_tca", action="store_true",
                    help="temporal credit assignment flag")
parser.add_argument("--power_iters", default=20, type=int,
                    help="Spectral normalization power iterations")
parser.add_argument("--dataset", default="celeba-hq",
                    help="Dataset name. MNIST, Omniglot, CelebA, CelebA-HQ is supported")

# Loss settings.
parser.add_argument("--entropy_cost", default=0.01, type=float,
                    help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=0.5, type=float,
                    help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.99, type=float,
                    help="Discounting factor.")

# Optimizer settings.
parser.add_argument("--policy_learning_rate", default=0.0003, type=float,
                    metavar="LRP", help="Policy learning rate.")
parser.add_argument("--discriminator_learning_rate", default=0.0001, type=float,
                    metavar="LRD", help="Discriminator learning rate.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")

# Misc settings.
parser.add_argument("--write_profiler_trace", action="store_true",
                    help="Collect and write a profiler trace "
                    "for chrome://tracing/.")

# yapf: enable


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)

frame_width = 64
grid_width = 32


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    entropy = 0
    for logit in logits:
        policy = F.softmax(logit, dim=-1)
        log_policy = F.log_softmax(logit, dim=-1)
        entropy += torch.sum(policy * log_policy)
    return entropy


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = 0
    for logit, action in zip(logits, actions):
        cross_entropy += F.nll_loss(
            F.log_softmax(torch.flatten(logit, 0, 1), dim=-1),
            target=torch.flatten(action.long(), 0, 1).squeeze(dim=-1),
            reduction="none",
        )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


def inference(flags, inference_batcher, model, image_queue, lock=threading.Lock()):
    with torch.no_grad():
        for batch in inference_batcher:
            batched_env_outputs, action, agent_state, image = batch.get_inputs()
            action = action.to(flags.actor_device, non_blocking=True)

            frame, _, done, *_ = batched_env_outputs
            frame = frame.to(flags.actor_device, non_blocking=True)
            done = done.to(flags.actor_device, non_blocking=True)

            if done.any().item():
                image_list = []
                for i in range(done.shape[1]):
                    image_list.append(image_queue.get())
                image = torch.stack(image_list, dim=1)

            if flags.condition:
                image = image.to(flags.actor_device)
                condition = image
            else:
                condition = None

            agent_state = nest.map(
                lambda t: t.to(flags.actor_device, non_blocking=True), agent_state,
            )

            with lock:
                T, B, *_ = frame.shape
                noise = torch.randn(T, B, 10).to(flags.actor_device, non_blocking=True)
                model = model.eval()
                outputs = model(
                    dict(
                        obs=frame,
                        condition=condition,
                        action=action,
                        noise=noise,
                        done=done,
                    ),
                    agent_state,
                )

            outputs = nest.map(lambda t: t.cpu(), outputs)
            core_output, core_state = outputs

            batch.set_outputs((core_output, core_state, noise, image.cpu()))


EnvOutput = collections.namedtuple(
    "EnvOutput", "frame, reward, done, episode_step episode_return"
)
AgentOutput = collections.namedtuple("AgentOutput", "action policy_logits baseline")
Batch = collections.namedtuple("Batch", "env agent")


def learn(
    flags,
    learner_queue,
    d_queue,
    model,
    actor_model,
    D,
    optimizer,
    scheduler,
    stats,
    plogger,
    lock=threading.Lock(),
):
    for tensors in learner_queue:
        tensors = nest.map(
            lambda t: t.to(flags.learner_device, non_blocking=True), tensors
        )

        batch, agent_state, image = tensors

        env_outputs, actor_outputs, noise = batch
        batch = (env_outputs, actor_outputs)
        frame, reward, done, *_ = env_outputs

        d_queue.put((frame, image.squeeze(0)))

        lock.acquire()  # Only one thread learning at a time.
        optimizer.zero_grad()

        actor_outputs = AgentOutput._make(actor_outputs)

        if flags.condition:
            condition = image
        else:
            condition = None

        model = model.train()
        learner_outputs, agent_state = model(
            dict(
                obs=frame,
                condition=condition,
                action=actor_outputs.action,
                noise=noise,
                done=done,
            ),
            agent_state,
        )

        if flags.use_tca:
            frame = torch.flatten(frame, 0, 1)
            if flags.condition:
                condition = torch.flatten(condition, 0, 1)
        else:
            frame = frame[-1]
            if flags.condition:
                condition = condition[-1]

        D = D.eval()
        with torch.no_grad():
            if flags.condition:
                p = D(frame, condition).view(-1, flags.batch_size)
            else:
                p = D(frame).view(-1, flags.batch_size)

            if flags.use_tca:
                d_reward = p[1:] - p[:-1]
                reward = reward[1:] + d_reward
            else:
                reward[-1] = reward[-1] + p
                reward = reward[1:]

            # empty condition
            condition = None

        # Take final value function slice for bootstrapping.
        learner_outputs = AgentOutput._make(learner_outputs)
        bootstrap_value = learner_outputs.baseline[-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = nest.map(lambda t: t[1:], batch)
        learner_outputs = nest.map(lambda t: t[:-1], learner_outputs)

        # Turn into namedtuples again.
        env_outputs, actor_outputs = batch

        env_outputs = EnvOutput._make(env_outputs)
        actor_outputs = AgentOutput._make(actor_outputs)
        learner_outputs = AgentOutput._make(learner_outputs)

        discounts = (~env_outputs.done).float() * flags.discounting

        action = actor_outputs.action.unbind(dim=2)

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=actor_outputs.policy_logits,
            target_policy_logits=learner_outputs.policy_logits,
            actions=action,
            discounts=discounts,
            rewards=reward,
            values=learner_outputs.baseline,
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs.policy_logits, action, vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs.baseline
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs.policy_logits
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        total_loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)

        optimizer.step()
        scheduler.step()

        actor_model.load_state_dict(model.state_dict())

        stats["step"] = stats.get("step", 0) + flags.unroll_length * flags.batch_size
        stats["total_loss"] = total_loss.item()
        stats["pg_loss"] = pg_loss.item()
        stats["baseline_loss"] = baseline_loss.item()
        stats["entropy_loss"] = entropy_loss.item()
        stats["final_reward"] = reward[-1].mean().item()
        stats["episode_reward"] = reward.mean(dim=1).sum().item()
        stats["learner_queue_size"] = learner_queue.size()

        if flags.condition:
            if flags.use_tca:
                _, C, H, W = frame.shape
                frame = frame.view(flags.unroll_length, flags.batch_size, C, H, W)
                frame = frame[-1]
            stats["l2_loss"] = F.mse_loss(frame, image.squeeze(0)).item()

        plogger.log(stats)
        lock.release()


real_label = 1.0
fake_label = 0.0


def learn_D(
    flags,
    queue,
    D,
    D_eval,
    optimizer,
    scheduler,
    stats,
    plogger,
    lock=threading.Lock(),
):
    while True:
        fake, real = nest.map(
            lambda t: t.to(flags.learner_device, non_blocking=True), queue.get()
        )

        if flags.condition:
            condition = real
        else:
            condition = None

        lock.acquire()
        optimizer.zero_grad()

        D = D.train()
        if flags.condition:
            p_real = D(real, condition).view(-1)
        else:
            p_real = D(real).view(-1)

        label = torch.full((flags.batch_size,), real_label, device=flags.learner_device)
        real_loss = F.binary_cross_entropy_with_logits(p_real, label)

        real_loss.backward()
        D_x = torch.sigmoid(p_real).mean()

        nn.utils.clip_grad_norm_(D.parameters(), flags.grad_norm_clipping)

        if flags.condition:
            T, *_ = fake.shape
            condition = condition.repeat(T, 1, 1, 1)

        fake = torch.flatten(fake, 0, 1)

        D = D.train()
        if flags.condition:
            p_fake = D(fake, condition).view(-1)
        else:
            p_fake = D(fake).view(-1)

        label.fill_(fake_label)
        fake_loss = F.binary_cross_entropy_with_logits(
            p_fake, label.repeat(flags.unroll_length + 1)
        )

        fake_loss.backward()
        D_G_z1 = torch.sigmoid(p_fake).mean()

        loss = real_loss + fake_loss

        nn.utils.clip_grad_norm_(D.parameters(), flags.grad_norm_clipping)

        optimizer.step()
        scheduler.step()

        D_eval.load_state_dict(D.state_dict())

        stats["D_loss"] = loss.item()
        stats["fake_loss"] = fake_loss.item()
        stats["real_loss"] = real_loss.item()
        stats["D_x"] = D_x.item()
        stats["D_G_z1"] = D_G_z1.item()

        lock.release()


def data_loader(
    flags, dataloader, image_queue,
):
    while True:
        for tensors in dataloader:
            if len(tensors) == 1:
                image = tensors
            elif len(tensors) <= 2:
                image = tensors[0]
            image_queue.put(image)


BRUSHES_BASEDIR = os.path.join(os.getcwd(), "third_party/mypaint-brushes-1.3.0")
BRUSHES_BASEDIR = os.path.abspath(BRUSHES_BASEDIR)

SHADERS_BASEDIR = os.path.join(os.getcwd(), "third_party/paint/shaders")
SHADERS_BASEDIR = os.path.abspath(SHADERS_BASEDIR)


def train(flags):
    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )

    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.learner_device = torch.device("cuda")
        flags.actor_device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.learner_device = torch.device("cpu")
        flags.actor_device = torch.device("cpu")

    if flags.max_learner_queue_size is None:
        flags.max_learner_queue_size = flags.batch_size

    # The queue the learner threads will get their data from.
    # Setting `minimum_batch_size == maximum_batch_size`
    # makes the batch size static.
    learner_queue = actorpool.BatchingQueue(
        batch_dim=1,
        minimum_batch_size=flags.batch_size,
        maximum_batch_size=flags.batch_size,
        check_inputs=True,
        maximum_queue_size=flags.max_learner_queue_size,
    )

    d_queue = Queue(maxsize=flags.max_learner_queue_size // flags.batch_size)
    image_queue = Queue(maxsize=flags.max_learner_queue_size)

    # The "batcher", a queue for the inference call. Will yield
    # "batch" objects with `get_inputs` and `set_outputs` methods.
    # The batch size of the tensors will be dynamic.
    inference_batcher = actorpool.DynamicBatcher(
        batch_dim=1,
        minimum_batch_size=1,
        maximum_batch_size=512,
        timeout_ms=100,
        check_outputs=True,
    )

    addresses = []
    connections_per_server = 1
    pipe_id = 0
    while len(addresses) < flags.num_actors:
        for _ in range(connections_per_server):
            addresses.append(f"{flags.pipes_basename}.{pipe_id}")
            if len(addresses) == flags.num_actors:
                break
        pipe_id += 1

    config = dict(
        episode_length=flags.episode_length,
        canvas_width=flags.canvas_width,
        grid_width=grid_width,
        brush_sizes=flags.brush_sizes,
    )

    if flags.dataset == "celeba" or flags.dataset == "celeba-hq":
        use_color = True
    else:
        use_color = False

    if flags.env_type == "fluid":
        env_name = "Fluid"
        config["shaders_basedir"] = SHADERS_BASEDIR
    elif flags.env_type == "libmypaint":
        env_name = "Libmypaint"
        config.update(
            dict(
                brush_type=flags.brush_type,
                use_color=use_color,
                use_pressure=flags.use_pressure,
                use_alpha=False,
                background="white",
                brushes_basedir=BRUSHES_BASEDIR,
            )
        )

    if flags.use_compound:
        env_name += "-v1"
    else:
        env_name += "-v0"

    env = env_wrapper.make_raw(env_name, config)
    if frame_width != flags.canvas_width:
        env = env_wrapper.WarpFrame(env, height=frame_width, width=frame_width)
    env = env_wrapper.wrap_pytorch(env)

    obs_shape = env.observation_space.shape
    if flags.condition:
        c, h, w = obs_shape
        c *= 2
        obs_shape = (c, h, w)

    action_shape = env.action_space.nvec.tolist()
    order = env.order
    env.close()

    model = models.Net(
        obs_shape=obs_shape,
        action_shape=action_shape,
        grid_shape=(grid_width, grid_width),
        order=order,
    )
    if flags.condition:
        model = models.Condition(model)
    model = model.to(device=flags.learner_device)

    actor_model = models.Net(
        obs_shape=obs_shape,
        action_shape=action_shape,
        grid_shape=(grid_width, grid_width),
        order=order,
    )
    if flags.condition:
        actor_model = models.Condition(actor_model)
    actor_model.to(device=flags.actor_device)

    D = models.Discriminator(obs_shape, flags.power_iters)
    if flags.condition:
        D = models.Conditional(D)
    D.to(device=flags.learner_device)

    D_eval = models.Discriminator(obs_shape, flags.power_iters)
    if flags.condition:
        D_eval = models.Conditional(D_eval)
    D_eval = D_eval.to(device=flags.learner_device)

    optimizer = optim.Adam(model.parameters(), lr=flags.policy_learning_rate)
    D_optimizer = optim.Adam(
        D.parameters(), lr=flags.discriminator_learning_rate, betas=(0.5, 0.999)
    )

    def lr_lambda(epoch):
        return (
            1
            - min(epoch * flags.unroll_length * flags.batch_size, flags.total_steps)
            / flags.total_steps
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    D_scheduler = torch.optim.lr_scheduler.LambdaLR(D_optimizer, lr_lambda)

    C, H, W = obs_shape
    if flags.condition:
        C //= 2
    # The ActorPool that will run `flags.num_actors` many loops.
    actors = actorpool.ActorPool(
        unroll_length=flags.unroll_length,
        learner_queue=learner_queue,
        inference_batcher=inference_batcher,
        env_server_addresses=addresses,
        initial_action=actor_model.initial_action(),
        initial_agent_state=actor_model.initial_state(),
        image=torch.zeros(1, 1, C, H, W),
    )

    def run():
        try:
            actors.run()
            print("actors are running")
        except Exception as e:
            logging.error("Exception in actorpool thread!")
            traceback.print_exc()
            print()
            raise e

    actorpool_thread = threading.Thread(target=run, name="actorpool-thread")

    c, h, w = obs_shape
    tsfm = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor()])

    dataset = flags.dataset

    if dataset == "mnist":
        dataset = MNIST(root="./", train=True, transform=tsfm, download=True)
    elif dataset == "omniglot":
        dataset = Omniglot(root="./", background=True, transform=tsfm, download=True)
    elif dataset == "celeba":
        dataset = CelebA(
            root="./", split="train", target_type=None, transform=tsfm, download=True
        )
    elif dataset == "celeba-hq":
        dataset = datasets.CelebAHQ(
            root="./", split="train", transform=tsfm, download=True
        )
    else:
        raise NotImplementedError

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, drop_last=True, pin_memory=True
    )

    stats = {}

    # Load state from a checkpoint, if possible.
    if os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath, map_location=flags.learner_device
        )
        model.load_state_dict(checkpoint_states["model_state_dict"])
        D.load_state_dict(checkpoint_states["D_state_dict"])
        optimizer.load_state_dict(checkpoint_states["optimizer_state_dict"])
        D_optimizer.load_state_dict(checkpoint_states["D_optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_states["D_scheduler_state_dict"])
        D_scheduler.load_state_dict(checkpoint_states["scheduler_state_dict"])
        stats = checkpoint_states["stats"]
        logging.info(f"Resuming preempted job, current stats:\n{stats}")

    # Initialize actor model like learner model.
    actor_model.load_state_dict(model.state_dict())
    D_eval.load_state_dict(D.state_dict())

    learner_threads = [
        threading.Thread(
            target=learn,
            name="learner-thread-%i" % i,
            args=(
                flags,
                learner_queue,
                d_queue,
                model,
                actor_model,
                D_eval,
                optimizer,
                scheduler,
                stats,
                plogger,
            ),
        )
        for i in range(flags.num_learner_threads)
    ]
    inference_threads = [
        threading.Thread(
            target=inference,
            name="inference-thread-%i" % i,
            args=(flags, inference_batcher, actor_model, image_queue,),
        )
        for i in range(flags.num_inference_threads)
    ]

    d_learner = [
        threading.Thread(
            target=learn_D,
            name="d_learner-thread-%i" % i,
            args=(flags, d_queue, D, D_eval, D_optimizer, D_scheduler, stats, plogger,),
        )
        for i in range(flags.num_learner_threads)
    ]
    for thread in d_learner:
        thread.daemon = True

    dataloader_thread = threading.Thread(
        target=data_loader, args=(flags, dataloader, image_queue,)
    )
    dataloader_thread.daemon = True

    actorpool_thread.start()

    threads = learner_threads + inference_threads
    daemons = d_learner + [dataloader_thread]

    for t in threads + daemons:
        t.start()

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "D_state_dict": D.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "D_optimizer_state_dict": D_optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "D_scheduler_state_dict": D_scheduler.state_dict(),
                "stats": stats,
                "flags": vars(flags),
            },
            checkpointpath,
        )

    def format_value(x):
        return f"{x:1.5}" if isinstance(x, float) else str(x)

    try:
        last_checkpoint_time = timeit.default_timer()
        while True:
            start_time = timeit.default_timer()
            start_step = stats.get("step", 0)
            if start_step >= flags.total_steps:
                break
            time.sleep(5)
            end_step = stats.get("step", 0)

            if timeit.default_timer() - last_checkpoint_time > 10 * 60:
                # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timeit.default_timer()

            logging.info(
                "Step %i @ %.1f SPS. Inference batcher size: %i."
                " Learner queue size: %i."
                " Other stats: (%s)",
                end_step,
                (end_step - start_step) / (timeit.default_timer() - start_time),
                inference_batcher.size(),
                learner_queue.size(),
                ", ".join(
                    f"{key} = {format_value(value)}" for key, value in stats.items()
                ),
            )
    except KeyboardInterrupt:
        pass  # Close properly.
    else:
        logging.info("Learning finished after %i steps.", stats["step"])
        checkpoint()

    # Done with learning. Stop all the ongoing work.
    inference_batcher.close()
    learner_queue.close()

    actorpool_thread.join()

    for t in threads:
        t.join()


def test(flags):
    if flags.xpid is None:
        checkpointpath = "./latest/model.tar"
    else:
        checkpointpath = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
        )

    config = dict(
        episode_length=flags.episode_length,
        canvas_width=flags.canvas_width,
        grid_width=grid_width,
        brush_sizes=flags.brush_sizes,
    )

    if flags.dataset == "celeba" or flags.dataset == "celeba-hq":
        use_color = True
    else:
        use_color = False

    if flags.env_type == "fluid":
        env_name = "Fluid"
        config["shaders_basedir"] = SHADERS_BASEDIR
    elif flags.env_type == "libmypaint":
        env_name = "Libmypaint"
        config.update(
            dict(
                brush_type=flags.brush_type,
                use_color=use_color,
                use_pressure=flags.use_pressure,
                use_alpha=False,
                background="white",
                brushes_basedir=BRUSHES_BASEDIR,
            )
        )

    if flags.use_compound:
        env_name += "-v1"
        config.update(
            dict(
                new_stroke_penalty=flags.new_stroke_penalty,
                stroke_length_penalty=flags.stroke_length_penalty,
            )
        )
    else:
        env_name += "-v0"

    env = env_wrapper.make_raw(env_name, config)
    if frame_width != flags.canvas_width:
        env = env_wrapper.WarpFrame(env, height=frame_width, width=frame_width)
    env = env_wrapper.wrap_pytorch(env)
    env = env_wrapper.AddDim(env)

    obs_shape = env.observation_space.shape
    if flags.condition:
        c, h, w = obs_shape
        c *= 2
        obs_shape = (c, h, w)

    action_shape = env.action_space.nvec.tolist()
    order = env.order

    model = models.Net(
        obs_shape=obs_shape,
        action_shape=action_shape,
        grid_shape=(grid_width, grid_width),
        order=order,
    )
    if flags.condition:
        model = models.Condition(model)
    model.eval()

    D = models.Discriminator(obs_shape, flags.power_iters)
    if flags.condition:
        D = models.Conditional(D)
    D.eval()

    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    D.load_state_dict(checkpoint["D_state_dict"])

    if flags.condition:
        from random import randrange

        c, h, w = obs_shape
        tsfm = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor()])
        dataset = flags.dataset

        if dataset == "mnist":
            dataset = MNIST(root="./", train=True, transform=tsfm, download=True)
        elif dataset == "omniglot":
            dataset = Omniglot(
                root="./", background=True, transform=tsfm, download=True
            )
        elif dataset == "celeba":
            dataset = CelebA(
                root="./",
                split="train",
                target_type=None,
                transform=tsfm,
                download=True,
            )
        elif dataset == "celeba-hq":
            dataset = datasets.CelebAHQ(
                root="./", split="train", transform=tsfm, download=True
            )
        else:
            raise NotImplementedError

        condition = dataset[randrange(len(dataset))].view((1, 1) + obs_shape)
    else:
        condition = None

    frame = env.reset()
    action = model.initial_action()
    agent_state = model.initial_state()
    done = torch.tensor(False).view(1, 1)
    rewards = []
    frames = [frame]

    for i in range(flags.episode_length - 1):
        if flags.mode == "test_render":
            env.render()
        noise = torch.randn(1, 1, 10)
        agent_outputs, agent_state = model(
            dict(
                obs=frame, condition=condition, action=action, noise=noise, done=done,
            ),
            agent_state,
        )
        action, *_ = agent_outputs
        frame, reward, done, _ = env.step(action)

        rewards.append(reward)
        frames.append(frame)

    reward = torch.cat(rewards)
    frame = torch.cat(frames)

    if flags.use_tca:
        frame = torch.flatten(frame, 0, 1)
        if flags.condition:
            condition = torch.flatten(condition, 0, 1)
    else:
        frame = frame[-1]
        if flags.condition:
            condition = condition[-1]

    D = D.eval()
    with torch.no_grad():
        if flags.condition:
            p = D(frame, condition).view(-1, 1)
        else:
            p = D(frame).view(-1, 1)

        if flags.use_tca:
            d_reward = p[1:] - p[:-1]
            reward = reward[1:] + d_reward
        else:
            reward[-1] = reward[-1] + p
            reward = reward[1:]

            # empty condition
            condition = None

    logging.info(
        "Episode ended after %d steps. Final reward: %.4f. Episode reward: %.4f,",
        flags.episode_length,
        reward[-1].item(),
        reward.sum(),
    )
    env.close()


def main(flags):
    if not flags.pipes_basename.startswith("unix:"):
        raise Exception("--pipes_basename has to be of the form unix:/some/path.")

    if flags.mode != "train":
        flags.start_servers = False

    if flags.start_servers:

        if flags.env_type == "fluid":
            env_name = "Fluid"
        elif flags.env_type == "libmypaint":
            env_name = "Libmypaint"

        if flags.use_compound:
            env_name += "-v1"
        else:
            env_name += "-v0"

        command = [
            "python",
            "-m",
            "torchbeast.polybeast_env",
            f"--num_servers={flags.num_actors}",
            f"--pipes_basename={flags.pipes_basename}",
            f"--env={env_name}",
            f"--env_type={flags.env_type}",
            f"--episode_length={flags.episode_length}",
            f"--canvas_width={flags.canvas_width}",
            f"--brush_sizes={flags.brush_sizes}",
            f"--new_stroke_penalty={flags.new_stroke_penalty}",
            f"--stroke_length_penalty={flags.stroke_length_penalty}",
        ]

        if flags.env_type == "fluid":
            assert flags.dataset != "omniglot" and flags.dataset != "mnist"

            command.extend(
                [f"--env={env_name}", f"--shaders_basedir={SHADERS_BASEDIR}"]
            )
        elif flags.env_type == "libmypaint":
            if flags.dataset == "celeba" or flags.dataset == "celeba-hq":
                command.append("--use_color")

            command.extend(
                [
                    f"--env={env_name}",
                    f"--brush_type={flags.brush_type}",
                    f"--background=white",
                    f"--brushes_basedir={BRUSHES_BASEDIR}",
                ]
            )

        if flags.use_pressure:
            command.append("--use_pressure")

        logging.info("Starting servers with command: " + " ".join(command))
        server_proc = subprocess.Popen(command)

    if flags.mode == "train":
        if flags.write_profiler_trace:
            logging.info("Running with profiler.")
            with torch.autograd.profiler.profile() as prof:
                train(flags)
            filename = "chrome-%s.trace" % time.strftime("%Y%m%d-%H%M%S")
            logging.info("Writing profiler trace to '%s.gz'", filename)
            prof.export_chrome_trace(filename)
            os.system("gzip %s" % filename)
        else:
            train(flags)
    else:
        test(flags)

    if flags.start_servers:
        # Send Ctrl-c to servers.
        server_proc.send_signal(signal.SIGINT)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    flags = parser.parse_args()
    main(flags)
