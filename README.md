# Adverserial Waddle Dynamics

A mallard version of ASE (https://xbpeng.github.io/projects/ASE/index.html)
![Skills](images/banner.png)

This repo unifies the training code using AMP for Open Duck Mini and the go_duck (real size version).

> **_NOTE:_**  The ASE part of this repo does not work yet. We only use the AMP implementation for now

## Installation

Download Isaac Gym from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions.

Once Isaac Gym is installed, install the external dependencies for this repo:

```
pip install -r requirements.txt
```


## AWD

[AMP](https://xbpeng.github.io/projects/AMP/index.html) works with reference motion. It can come from motion capture, or from some kind of reference motion generator.

We use [placo](https://github.com/Rhoban/placo) to generate parametric gait reference motion.

See [this](gait_generation/README.md) to generate some reference motion. To train a forward only walking motion, only one example of walking forward is enough. For a holonomous walk, generating ~100 random trajectories seems to be enough.

### Training

#### Relevent source and config files

These are the files that you are most likely to edit:

In `awd/data/`
- `assets/` contains the URDF and related mesh files
  - `assets/<robot>/urdf/<robot>_props.yaml` contains actuator and joints properties
- `cfg/` contains the configuration files for the different tasks, for each robot.
  - `<robot>/duckling_command.yaml` contains the configuration of the `command` task
  - `<robot>/train/amp_duckling_task.yaml` contains the training configuration and hyperparameters
- `motion/` contains the reference motion files (`.json`)

In `awd/env/tasks/`
- `awd/env/tasks/duckling.py`
- `awd/env/tasks/duckling_command.py`
- `awd/env/tasks/duckling_amp.py`
- `awd/env/tasks/duckling_amp_task.py`

We mainly use the command task, in which the agent is trained to track velocities, as well as following the reference motion style using AMP.

Train using the following command:
```bash
python awd/run.py --task DucklingCommand --num_envs <...> --cfg_env awd/data/cfg/<robot>/duckling_command.yaml --cfg_train awd/data/cfg/<robot>/train/amp_duckling_task.yaml --motion_file awd/data/motions/<robot>/
```

`--motion_file` can be used to specify a dataset of motion clips that the model should imitate. You can specify a specific file or a directory containing multiple files.

Checkpoints are saved in `output/`.

To test a trained model, use the following command:
```bash
python awd/run.py --test --task DucklingCommand --num_envs <...> --cfg_env awd/data/cfg/<robot>/duckling_command.yaml --cfg_train awd/data/cfg/<robot>/train/amp_duckling_task.yaml --motion_file awd/data/motions/<robot>/ --checkpoint <path_to_checkpoint.pth>
```

&nbsp;

### Viewing URDF

```
python view_urdf.py awd/data/assets/go_bdx/go_bdx.urdf
```

```
python view_urdf.py awd/data/assets/mini_bdx/urdf/bdx.urdf
```

### Viewing Placo Frames

```
python view_urdf.py awd/data/assets/go_bdx/go_bdx.urdf --frames left_foot right_foot trunk head
```

```
python view_urdf.py awd/data/assets/mini_bdx/urdf/bdx.urdf --frames left_foot right_foot trunk head
```
