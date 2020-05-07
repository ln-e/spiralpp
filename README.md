
# SPIRAL++

A PyTorch implementation of [Unsupervised Doodling and Painting with Improved SPIRAL
by Mellor, Park, Ganin et al.](https://arxiv.org/abs/1802.01561)

For further details, see https://learning-to-paint.github.io for paper with generation videos.

### Installing

#### Linux

Create a new Conda environment, and install PolyBeast's requirements:

```shell
$ conda create -n spiralpp python=3.7
$ conda activate spiralpp
$ pip install -r requirements.txt
```

Install spiral-gym

Install required packages:

```shell
$ apt-get install cmake pkg-config protobuf-compiler libjson-c-dev intltool
$ pip install six setuptools numpy scipy gym
```

**WARNING:** Make sure that you have `cmake` **3.14** or later since we rely
on its capability to find `numpy` libraries.

Install cmake by running:
```shell
$ conda install cmake
```

Finally, run the following command to install the spiral-gym package itself:

```shell
$ git submodule update --init --recursive
$ pip install -e spiral_gym/
```

You will also need to obtain the brush files for the `libmypaint` environment
to work properly. These can be found
[here](https://github.com/mypaint/mypaint-brushes). For example, you can
place them in `third_party` folder like this:

```shell
$ wget -c https://github.com/mypaint/mypaint-brushes/archive/v1.3.0.tar.gz -O - | tar -xz -C third_party
```

Finally, the `Fluid Paint` environment depends on the shaders from the original
`javascript` [implementation](https://github.com/dli/paint). You can obtain
them by running the following commands:

```shell
$ git clone https://github.com/dli/paint third_party/paint
$ patch third_party/paint/shaders/setbristles.frag third_party/paint-setbristles.patch
```

PolyBeast requires installing PyTorch
[from source](https://github.com/pytorch/pytorch#from-source).

PolyBeast also requires gRPC, which can be installed by running:

```shell
$ conda install -c anaconda protobuf
$ ./scripts/install_grpc.sh
```

Compile the C++ parts of PolyBeast:

```
$ pip install nest/
$ export LD_LIBRARY_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib:${LD_LIBRARY_PATH}
$ python setup.py install
```

### Running PolyBeast

To start both the environment servers and the learner process, run
```shell
$ python -m torchbeast.monobeast \
     --dataset celeba-hq \
     --env_type libmypaint \
     --canvas_width 64 \
     --use_pressure \
     --use_tca \
     --power_iters 40 \
     --num_actors 64 \
     --total_steps 30000000 \
     --learning_rate 0.0004 \
     --entropy_cost 0.01 \
     --batch_size 64 \
     --episode_length 40 \
     --xpid example
```

Results are logged to `~/logs/torchbeast/latest` and a checkpoint file is
written to `~/logs/torchbeast/latest/model.tar`.

The environment servers can also be started separately:

```shell
$ python -m torchbeast.polybeast_env --num_servers 10
```

Start another terminal and run:

```shell
$ python -m torchbeast.polybeast --no_start_servers
```

## (Very rough) overview of the system

```
|-----------------|     |-----------------|                  |-----------------|
|     ACTOR 1     |     |     ACTOR 2     |                  |     ACTOR n     |
|-------|         |     |-------|         |                  |-------|         |
|       |  .......|     |       |  .......|     .   .   .    |       |  .......|
|  Env  |<-.Model.|     |  Env  |<-.Model.|                  |  Env  |<-.Model.|
|       |->.......|     |       |->.......|                  |       |->.......|
|-----------------|     |-----------------|                  |-----------------|
   ^     I                 ^     I                              ^     I
   |     I                 |     I                              |     I Actors
   |     I rollout         |     I rollout               weights|     I send
   |     I                 |     I                     /--------/     I rollouts
   |     I          weights|     I                     |              I (frames,
   |     I                 |     I                     |              I  actions
   |     I                 |     v                     |              I  etc)
   |     L=======>|--------------------------------------|<===========J
   |              |.........      LEARNER                |
   \--------------|..Model.. Consumes rollouts, updates  |    Learner       |----------------------------|
     Learner      |.........       model weights         |     sends        |    DISCRIMINATOR LEARNER   |
      sends       |                                      |    weights       |.................           |
     weights      |.................                     |<=================|..Discriminator..           |
                  |..Discriminator.. Computes reward     |----------------->|.................           |
                  |.................                     |  Learner sends   | Consumes frames and images |
                  |--------------------------------------|  frames, images  |----------------------------|
```

The system has three main components, actors, learner and d_learner.

Actors generate rollouts (tensors from a number of steps of
environment-agent interactions, including environment frames, agent
actions and policy logits, and other data).

The learner consumes that experience, computes the reward using the discriminator, 
computes a loss and updates the weights. The new weights are then propagated to the actors.
Frame and image pairs are sent to d_learner.

D_learner consumes that pair and computes a loss and updates the weights.
The new weights are then propagated to the learner's discriminator.

## Repository contents

`libtorchbeast`: C++ library that allows efficient learner-actor
communication via queueing and batching mechanisms. Some functions are
exported to Python using pybind11. For PolyBeast only.

`nest`: C++ library that allows to manipulate complex
nested structures. Some functions are exported to Python using
pybind11.

`third_party`: Collection of third-party dependencies as Git
submodules. Includes [gRPC](https://grpc.io/), .

`torchbeast`: Contains `monobeast.py`, and `polybeast.py` and
`polybeast_env.py`. (`monobeast.py` is currently unavailable)

`spiral-gym`: libmypaint and fluidpaint based environments. ported to openai
gym from https://github.com/deepmind/spiral/tree/master/spiral/environments

## TODO
- [x] environments with compound action space 
- [x] environments with penalty on stroke length and new stroke
- [ ] population based training (no plans as of now)
- [ ] python tests like the original torchbeast has

## License

spiralpp is released under the Apache 2.0 license.
