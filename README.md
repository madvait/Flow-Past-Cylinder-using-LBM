# Lattice Boltzmann Model (LBM) Simulation
This repository contains Python code for simulating fluid flow past a circular cylinder using the Lattice Boltzmann Method (LBM). The LBM is a powerful computational fluid dynamics technique that models fluid behavior on a lattice grid.

## Installation
To run the code, you need to have Python installed. Additionally, you'll need the following libraries:
* '```numpy```'
* '```matplotlib```'
* '```tqdm```'

  You can install these libraries using pip:
  ``` pip install numpy matplotlib tqdm```

## Usage

The main script for running the simulation is ```lbm_simulation.py```. You can run it using python:
  ``` python lbm_simulation.py```

The simulation parameters such as grid size, inflow velocity, and number of iterations can be adjusted in the script.

## Files
* `lbm.py`: Contains the core LBM functions for calculating density, velocity, and equilibrium distribution.
* `lbm_simulation.py`: Main script for running the LBM simulation.
* `README.md`: This file, contains information about the project.

## Notes:

Additional notes about the theory can be found in the Jupyter Notebook: `notes_on_lbm_cylinder.ipynb`





