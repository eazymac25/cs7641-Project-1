# CS7641-Project-1
Project 1

Repository: https://github.com/eazymac25/cs7641-Project-1

## Data Sets

- [US Census Data]()
- [Wine Acidity]()

## Classifiers
- [Decision Tree with pruning]()
- [Multi-Layer Neural Network]()
- [Boosted Decision Tree]()
- [Support Vector Machine]()
- [KNN]()

## Directory Structure
```
|-- exploratory_notebooks
|-- solution
    |-- classifiers
        |-- census_output (census output graphs)
        |-- wine_output (wine output graphs)
        |-- <all_models>.py
        |-- times.txt (time log)
    |-- data
        |-- raw_census_data.csv
        |-- winequality-red.csv
    |-- preprocessors
        |-- census_histograms (histograms of features)
        |-- wine_histograms
        |-- data_loader.py (data pipeline to run preprocessing)
```

## Installation

### Directions
1. Install [Anaconda](https://www.anaconda.com/)
    - please download the anaconda appropriate for your OS.
    - **Version**: Python 3.7.1

2. Install graphviz (Mac or Linux)

    ```bash
    # Linux
    sudo apt-get install graphviz
    # Mac 
    sudo brew install graphviz
    ```

3. Download project

    - [Google Drive]()
    - `git clone https://github.com/eazymac25/cs7641-Project-1.git`

4. Move into directory

    - `cd cs764l-Project-1`
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