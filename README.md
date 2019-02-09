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
    
5. Install Requirements (make sure you did step 4)

    ```bash
    conda create --name kmacneney3-solution python=3.7
    conda activate kmacneney3-solution
    conda install --file requirements.txt
    ```
    Or if you don't care about using an virtual environment
    ```bash
    pip install -r requirements.txt
    # or with conda
    conda install --file requirements.txt
    ```