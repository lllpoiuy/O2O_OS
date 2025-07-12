# O2O_OS
offline-to-online RL operating system

## TODO

### Algorithm support

Algorithms
+ offline-RL based
  + iql
  + cql
  + cal-ql
  + rlpd
  + wsrl
+ diffusion-based methods
  + dppo
  + dsrl
  + fql
  + qc-fql
+ vla-based methods
  + rldg
  + pa-rl
+ imitation-based methods
  + ibrl
  + dgn

Main components:
+ critic_loss
+ actor_loss 
+ sample_action
+ sample_batch
+ create_networks
+ imitation controller
+ offline controller
+ warm-up controller
+ online controller
+ dataset & env

Plugin-compenents:
+ expectile regression (critic_loss)
+ conservatism term (critic_loss)
+ calibrated term (critic_loss)
+ auxiliary one-step denoiser (sample_action, actor_loss, create_networks)
+ action chunking (critic_loss, actor_loss, create_networks)
+ best-of-n sampling (sample_action)
+ action gradient (sample_action)
+ base policy bootstraped exploration (sample_action)
+ noise guided exploration (sample_action)
+ auxiliary normal learner (actor_loss, sample_action)

Networks:
+ Q-ensemble
+ layernorm critic
+ normal actor
+ flow matching actor
+ diffusion actor
+ vla actor


## Acknowledgement


Repos:
```
rlpd
https://arxiv.org/pdf/2302.02948
https://github.com/ikostrikov/rlpd

cal-ql
https://arxiv.org/pdf/2303.05479
https://github.com/nakamotoo/Cal-QL

wsrl
https://arxiv.org/pdf/2412.07762
https://github.com/zhouzypaul/wsrl

dppo
https://arxiv.org/pdf/2409.00588
https://github.com/irom-princeton/dppo

diffusion steering
https://arxiv.org/pdf/2506.15799
https://diffusion-steering.github.io/

policy decorator
https://arxiv.org/pdf/2412.13630
https://github.com/tongzhoumu/policy_decorator

pa-rl
https://arxiv.org/pdf/2412.06685
https://github.com/MaxSobolMark/PolicyAgnosticRL

qc
https://papers.cool/arxiv/2507.07969
http://github.com/ColinQiyangLi/qc
```

Papers:


**IQL** (Sergey, 2021): Offline Reinforcement Learning with Implicit Q-Learning, https://arxiv.org/pdf/2110.06169

**CQL** (Sergey, 2020); Conservative Q-Learning for Offline Reinforcement Learning, https://arxiv.org/pdf/2006.04779

$\star$**Cal-QL** (nips, Sergey, 2023): Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning, https://arxiv.org/pdf/2303.05479

$\star$**RLPD** (Sergey, 2023): Efficient Online Reinforcement Learning with Offline Data, https://arxiv.org/pdf/2302.02948



**DPPO** (2024); Diffusion Policy Policy Optimization, https://arxiv.org/pdf/2409.00588

**FQL** (Sergey, 2025); Flow Q-Learning, https://arxiv.org/abs/2502.02538

**DSRL** (Sergey, 2025); Steering Your Diffusion Policy with Latent Space Reinforcement Learning, https://arxiv.org/pdf/2506.15799

**QC** (Sergey, 2025); Reinforcement Learning with Action Chunking, https://papers.cool/arxiv/2507.07969



**IBRL** (Stanford, 2023); Imitation Bootstrapped Reinforcement Learning, https://arxiv.org/pdf/2311.02198

**PA-RL** (Finn, 2024); Policy Agnostic RL: Offline RL and Online RL Fine-Tuning of Any Class and Backbone, https://arxiv.org/pdf/2412.06685

**RLDG** (Sergey, 2025): Robotic Generalist Policy Distillation via Reinforcement Learning, https://generalist-distillation.github.io/static/high_performance_generalist.pdf



**WSRL** (Sergey, 2024); Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data, https://arxiv.org/pdf/2412.07762

**DGN** (Finn, 2025); Reinforcement Learning via Implicit Imitation Guidance, https://arxiv.org/pdf/2506.07505v1


## Environment Setup

```bash
conda create -n jaxrl python=3.10

# in dppo
pip install -e .
pip install -e .[kitchen]
conda install -c conda-forge glew mesalib
conda install -c menpo glfw3

# jax
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install --upgrade "jax[cuda12_local]"==0.5.0 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# cal-ql
pip install distrax chex patchelf moviepy flax==0.10.2 ml_collections typing-extensions optax tqdm

# rlpd
pip install lockfile
git clone https://github.com/ikostrikov/dmcgym
cd ./dmcgym
# remove gym[mujoco] from /dmcdym/requirements.txt
pip install -e .
cd ../
# replace all "tfp = tensorflow_probability.substrates.jax" with "import tensorflow_probability.substrates.jax as tfp"

# wsrl
pip install pre-commit overrides
git clone --recursive https://github.com/nakamotoo/mj_envs.git
cd mj_envs
git submodule update --remote
pip install -e .
git clone https://github.com/aravindr93/mjrl.git
cd mjrl
pip install -e .

# pa-rl
pip install plotly tensorflow==2.19.0 opencv-python seaborn transformers tf_keras==2.19.0

# qc
pip install ogbench opt_einsum pillow platformdirs opt_einsum pillow platformdirs protobuf psutil Pygments PyOpenGL six simplejson gymnasium ogbench
conda install cmake
pip install robomimic

```