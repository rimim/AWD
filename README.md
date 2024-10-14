# Adverserial Waddle Dynamics

A mallard version of ASE (https://xbpeng.github.io/projects/ASE/index.html)
![Skills](images/banner.png)

### Installation

Download Isaac Gym from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions.

Once Isaac Gym is installed, install the external dependencies for this repo:

```
pip install -r requirements.txt
```


### AWD

#### Pre-Training

First, an AWD model can be trained to imitate a dataset of motions clips using the following command:
```
python awd/run.py --task DucklingAMP --cfg_env awd/data/cfg/go_bdx/duckling.yaml --cfg_train awd/data/cfg/go_bdx/train/awd_duckling.yaml --motion_file awd/data/motions/go_bdx/walk_forward.txt --headless
```
`--motion_file` can be used to specify a dataset of motion clips that the model should imitate.
The task `GoBDX` will train a model to imitate a dataset of motion clips.
Over the course of training, the latest checkpoint `Checkpoint.pth` will be regularly saved to `output/`,
along with a Tensorboard log. `--headless` is used to disable visualizations. If you want to view the
simulation, simply remove this flag. To test a trained model, use the following command:
```
python awd/run.py --test --task DucklingAMP --num_envs 8 --cfg_env awd/data/cfg/go_bdx/duckling.yaml --cfg_train awd/data/cfg/go_bdx/train/awd_duckling.yaml --motion_file awd/data/motions/go_bdx/walk_forward.txt --checkpoint [path_to_awd_checkpoint]
```
You can also test the robustness of the model with `--task GoBDXPerturb`, which will throw projectiles at the character.

&nbsp;

&nbsp;

### AMP

We also provide an implementation of Adversarial Motion Priors (https://xbpeng.github.io/projects/AMP/index.html).
A model can be trained to imitate a given reference motion using the following command:
```
python awd/run.py --task DucklingAMP --cfg_env awd/data/cfg/go_bdx/train/amp_duckling.yaml --cfg_train awd/data/cfg/go_bdx/train/amp_duckling.yaml --motion_file awd/data/motions/go_bdx/walk_forward.txt --headless
```
The trained model can then be tested with:
```
python awd/run.py --test --task DucklingAMP --num_envs 16 --cfg_env awd/data/cfg/go_bdx/train/amp_duckling.yaml --cfg_train awd/data/cfg/go_bdx/train/amp_duckling.yaml --motion_file awd/data/motions/go_bdx/walk_forward.txt --checkpoint [path_to_amp_checkpoint]
```

&nbsp;

&nbsp;

### Motion Data

Motion clips are located in `awd/data/motions/`. Individual motion clips are stored as `.json` files. Motion datasets are specified by `.yaml` files, which contains a list of motion clips to be included in the dataset. Motion clips can be visualized with the following command:
```
python awd/run.py --test --task DucklingViewMotion --num_envs 2 --cfg_env awd/data/cfg/go_bdx/duckling.yaml --cfg_train awd/data/cfg/go_bdx/train/amp_duckling.yaml --motion_file awd/data/motions/go_bdx/walk_forward.json
```

```
python awd/run.py --test --task DucklingViewMotion --num_envs 2 --cfg_env awd/data/cfg/mini_bdx/duckling.yaml --cfg_train awd/data/cfg/mini_bdx/train/amp_duckling.yaml --motion_file awd/data/motions/mini_bdx/walk_forward.json
```
`--motion_file` can be used to visualize a single motion clip `.npy` or a motion dataset `.yaml`.


### Gait Generation

See [this](gait_generation/README.md)

#### Gait Playground

```
python gait_playground.py
```

```
python gait_playground.py --mini
```

#### Viewing URDF

```
python view_urdf.py awd/data/assets/go_bdx/go_bdx.urdf
```

```
python view_urdf.py awd/data/assets/mini_bdx/urdf/bdx.urdf
```

#### Viewing Placo Frames

```
python view_urdf.py awd/data/assets/go_bdx/go_bdx.urdf --frames left_foot right_foot trunk head
```

```
python view_urdf.py awd/data/assets/mini_bdx/urdf/bdx.urdf --frames left_foot right_foot trunk head
```
