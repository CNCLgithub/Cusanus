# Ensembles

Ensembles powered by normalizing flow

## Setup and running
1. Clone repository `git clone https://github.com/CNCLgithub/Ensembles` and `cd Ensembles`
2. Get deps using `git submodule update --init --recursive`
2. Run `./env.d/setup.sh cont_pull python julia` to build the container and setup python and Julia.
3. Enter `./env.d/run.sh julia` to get into Julia REPL

This project has automatic configuration!! This configuration is defined in `default.conf`.
You should always prepend `./run.sh` before any command (including running programs like `julia`) to ensure consistency. 
If you wish to have different values than `default.conf`, simply:

``` sh
cp default.conf user.conf
vi user.conf # edit to your liking without adding new elements
```
### Mac and Window users

In order to use singularity you must have a virtual machine running. 
Assuming you have vagrant (and something like virtualbox) setup on your host, you can follow these steps

## Contributing

### Contributing Rules


1. Place all re-used code in packages (`src`)
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

