

# Installation

NOTE: The installation relies on Anaconda.

### Prerequisites
 - [Anaconda](https://www.anaconda.com/)
    - please download the anaconda appropriate for your OS.

```bash
# make sure to install graphviz
sudo apt-get install graphviz
# or for Macs
sudo brew install graphviz

git clone https://github.com/eazymac25/cs7641-Project-1.git

cd cs7641-Project-1

conda create --name myenv python=3.7 # creates full env

# or

conda env create -f environment.yml

# activate the env
conda activate myenv

# skip if you used the environment.yml
conda install --file requirements.txt

```