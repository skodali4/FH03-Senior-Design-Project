## Setup
 
For an identical setup, build a virtual env and install requirements.txt.
I recommend using conda as that will ensure the smoothest install for cyipopt (from what I know)

## Conda Env:

#### Windows /(Macos??)
I only tested this with windows so I'm actually not sure if it will work on mac, but lmk if it does :)

Update: might need to delete or change the "prefix" in .yml file

Try the following
`conda env create -f environment.yml` <br>
`conda activate testenv` <br>
`python run_sim.py`

If these work for you it will build the conda environment and name it testenv. You can add to 
it locally as needed. If you'd like to change the name of the environment you can open the .yml
file and edit the first line to whatever name you'd like to use

#### Manual 
If that does not work, unfortunately you'll probably need to set up more manually. You can try:

`conda create --name your_env_name`
`conda activate your_env_name`
`pip install -r requirements.txt`
`conda install -c conda-forge cyipopt`
`python run_sim.py`

NOTE: cyipopt will have to be installed separately here as from what I know pip will break. The command
provided is their recommended install method


## Usage
You can configure some settings by editing the run_sim.py file. 

Or just build the plots by running:
`python run_sim.py -o plot'

Note that animate.py, stats_plots.py will NOT run on their own until run_sim.py
populates results to "data/" folder.

