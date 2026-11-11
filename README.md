# FRB-flow
<img width="3300" height="1200" alt="FRB20190122C(2)" src="https://github.com/user-attachments/assets/a37c1097-7135-4f68-92bd-e589ce1bc43a" />

Source code for project on flow matching for trans-dimensional simulation-based inference on fast astronomical transients. 

Read the associated paper (pre-print) here:<br>
📄[Fast Inference on Astronomical Time Series with Trans-Dimensional Flow Matching Posterior Estimation](https://arxiv.org/abs/2607.21134)

## Installation 

To get started, clone the repository 
```bash
git clone https://github.com/Ninavd/FRB-flow.git
cd FRB-flow
```

recreate the conda environment using the environment file
```
conda env create --file environment.yml
```

## Repository structure
```
└── MCMC_runs/             # MCMC run results are stored here
|    └── ...
├── notebooks/
│   ├── FRB_prep/          # Notebooks for processing raw FRB dynamic spectra      
│   │   └── ...                   
│   ├── ...                # Notebooks for evaluating trained FM models    
├── scripts/         
|    ├── FM_training.py    # training an FM model via the CLI  
|    └── MCMC.py           # running MCMC sampling via the CLI
├── observational_data/
|    ├── FRBs/             # pre-processed FRB profiles
|    └── magnetars/        # raw magnetar bursts time series
├── src/             
|    └── flow_matching/
|         ├── distributions.py
|         ├── helpers.py
|         ├── integration.py
|         ├── loader.py
|         ├── models.py
|         ├── neural_posterior.py
|         ├── plotting.py
|         ├── probability_path.py
|         ├── simulator.py
|         ├── training.py
|         └── transformer.py
|    └── MCMC/
|        ├── helpers.py
│        ├── plotting.py
│        ├── posterior.py
│        └── priors.py 
├── c2st.py 
├── helpers.py 
├── simulator.py 
```
