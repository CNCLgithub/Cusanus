# Event-driven Simulation

A new mode of mental simulation for Intuitive Physics

## Setup and running

1. Clone repository and cd into root directory
2. Get any deps using `git submodule update --init --recursive` (could be empty)
3. Download the [container](https://drive.google.com/uc?export=download&id=1Cw1BDlvIyE8thDVfpMfHL9khGxBKtHoX) and place under `env.d`
4. Run `./env.d/setup.sh all` to setup the environment.
5. Enter `./env.d/run.sh COMMAND` to run command through the project environment (e.g., python script or julia REPL)

This project has automatic configuration. This configuration is defined in `env.d/default.conf`.
You should always prepend `env.d/run.sh` before any command (including running programs like `julia`) to ensure consistency. 
If you wish to have different values than `default.conf`, simply:

``` sh
cp default.conf user.conf
vi user.conf # edit to your liking without adding new elements
```
### Mac and Window users

In order to use apptainer/singularity, please refer to the official instructions: http://apptainer.org/docs/admin/latest/installation.html#installation-on-windows-or-mac

## Contributing

### Contributing Rules


1. Place all re-used code in packages (`src` or `cusanus`)
2. Place all interactive code in `scripts`
3. Do not use "hard" paths. Instead refer to the paths in `SPATHS`.
4. Add contributions to branches derived from `master` or `dev`
4. Avoid `git add *`
5. Do not commit large files (checkpoints, datasets, etc). Update `setup.sh` accordingly.


### Project layout

The python package environment is managed by as defined in `setup.sh` (specifically `SENV[pyenv]`)
Likewise, the Julia package is described under `src` and `test`

All scripts are located under `scripts` and data/output is under `env.d/spaths` as specific in the project config (`default.conf` or `user.conf`)


### Changing the enviroment

To add new python or julia packages use the provided package managers ( `Pkg.add ` for julia )
For julia you can also use `] add ` in the REPL
For python, add things to requirements.txt or `setup.py` if one is present

