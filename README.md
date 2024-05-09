# voltage_mining_model_demo

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Premise

This is a demonstration module for a manuscript titled "Voltage mining for delithiation-stabilized cathodes and a machine-learning model for Li-ion cathode voltage" (see last section for citation). The purpose of this demonstration module is to provide a way for those inserested to use the LIB cathode voltage prediction model constructed in the manuscript. For details and limits of the model, please see the manuscript.

## Installation

To get started, navigate to a directory where you want to install this repo, and clone this repo:

```bash
git clone https://github.com/hmlli/voltage_mining_model_demo.git
```

The main functionalies of this demonstration model lives in the `voltage_mining_model_demo.ipynb` jupyter notebook. To run this file, you need the following software:
```
python==3.10.14
jupyterlab==4.1.5
ipykernel==6.29.3
xgboost==2.0.3
joblib==1.3.2
matminer==0.9.0
scikit-learn==1.4.1.post1
pymatgen==2023.9.25
numpy==1.26.4
pandas==1.5.3
```
and their dependencies.

Make sure you have conda installed and start a new virtual environment to run this demo module:

```bash
conda create -n vmm_demo python=3.10.14
```

Then activate the environment and navigate to the repo directory:
```bash
conda activate vmm_demo
cd voltage_mining_model_demo
```

Then navigate to the repo directory and install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

or manually with:
```bash
pip install jupyterlab==4.1.5
pip install xgboost==2.0.3
pip install joblib==1.3.2
pip install matminer==0.9.0
pip install pymatgen==2023.9.25
```
All other softwares should have been installed with the installation of the above. If you run into uninstall packages or version issues, please follow the list at the top to ensure correct software and versions.

Then open the `votlage_mining_model_demo.ipynb` file with a jupyter-compatible IDE or open it in your browser:

```bash
jupyter lab
```

and proceed to the instructions in the notebook.

## Citation

Please cite the following work where this model is constructed if you use this code: [[arXiv](placeholder)]

`arXiv:`
```tex
placeholder
```

`publication:`
```tex
placeholder
```