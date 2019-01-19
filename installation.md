

# Installation

```bash
git clone https://myrepo.com

cd myrepo

conda create --name myenv python=3.7 # creates full env

# or

conda env create -f environment.yml

# activate the env
conda activate myenv

# skip if you used the environment.yml
conda install --file requirements.txt

```