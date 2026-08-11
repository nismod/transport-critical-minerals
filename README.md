# Transport Optimisation Model for Critical Minerals Resource Allocation
This repository solves a spatial transport flow optimisation problem to estimate locations of mineral processing plants. It has been applied to 6 critical minerals (Cobalt, Copper, Graphite, Lithium, Manganese and Nickel) in 14 African countries. The analysis focusses on spatially mapping transport flows from mines to road and railway networks in African and onwards to maritime shipping networks globally. The spatial optimisation finds future locations of processing plants based on a maximisation of flow aggregation and a minisation of the transport costs. The model also solves for constraints on accessibility to the energy grid, avoidance of pristine forests, protected land-use areas, key biodiversity areas, high to very high water stress areas and arid areas.

# Key references
This method is explained in detail in:

Pant, R., Cervantes Barron, K., Andrieu, B., Foster, V., Hall, J.W. 2026. Battery Mineral Value Chains in Africa: Transport optimization model for critical minerals resource allocation. Forthcoming

The method has been used in:
Cervantes Barron, K., Pant, R., Andrieu, B., Heydari, M., Korkovelos, A., Osei-Owusu, S., Barzin, S., Ciftci, M.M., Stringer, M., Gomez, C.R., Hawkes, A., Hall, J., Foster, V., 2026. Nationalist versus regional approaches for increased battery mineral value addition in Southern Africa. https://doi.org/10.21203/rs.3.rs-9179870/v1 (Under review in Nature Communications)

# Setup and installation
Clone or download this repository from
[GitHub](https://github.com/nismod/transport-critical-minerals):

    git clone git@github.com:nismod/transport-critical-minerals.git

Next, install required Python packages:

We recommend using [`micromamba`](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)
to install the packages and manage installing libraries into a conda
environment, usually handling non-Python dependencies well.

Create a conda environment once (per machine/user):

    micromamba create --file environment.yaml

# Configuration
Copy the template and set your local paths:

	cp config.template.json config.json


# Running the analysis
Full analysis is executed by running the script:

	python scripts/updated_setup/combination_setup.py

Sample plotting scripts are in the folder:

	scripts/plot

# Funding support
This work is supported by the Climate Compatible Growth (CCG) project funded by the UK Foreign, Commonwealth and Development Office (FCDO) Grant Number [GB-GOV-1-300125](https://devtracker.fcdo.gov.uk/programme/GB-GOV-1-300125/summary)   