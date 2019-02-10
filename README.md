# CS7641-Project-1
Project 1

Author: Kyle MacNeney

Author Email: kyle.macneney@gmail.com

Repository: https://github.com/eazymac25/cs7641-Project-1

*Please contact for access to the repository* 

**NOTE: Written in markdown**

## Data Sets

- [US Census Data](https://www.kaggle.com/uciml/adult-census-income)
- [Wine Quality](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009)

## Classifiers
- [Decision Tree with pruning](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Multi-Layer Neural Network](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [Boosted Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
- [Support Vector Machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

## Directory Structure
```
|-- exploratory_notebooks (some initial data exploration)
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
|-- README.md (markdown readme)
|-- README.txt (text readme)
|-- requirements.txt
|-- run.py (python command line which allows running each model)
```

## Installation

1. Install [Anaconda](https://www.anaconda.com/)
    - please download the anaconda appropriate for your OS.
    - **Version**: Python 3.7.1

2. Install graphviz (Mac or Linux only)

    ```bash
    # Linux
    sudo apt-get install graphviz
    # Mac 
    sudo brew install graphviz
    ```

3. Download project

    - [Google Drive](https://drive.google.com/file/d/1n6YWpt0A9GJQEYdeehC8o6DHYNy9kTYz/view?usp=sharing)
    - `git clone https://github.com/eazymac25/cs7641-Project-1.git`

4. Move into directory
    - If downloaded the zip from drive: `unzip kmacneney3-project1-soluition.zip && cd cs7641-Project-1`
    - If cloned from git: `cd cs764l-Project-1`
    
5. Install Requirements (make sure you did step 4)

    ```bash
    conda create --name kmacneney3-solution python=3.7
    conda activate kmacneney3-solution
    conda install --file requirements.txt
    ```
    Or if you don't care about using a virtual environment
    ```bash
    pip install -r requirements.txt
    # or with conda
    conda install --file requirements.txt
    ```
    
## Running Experiments

```bash
python run.py [census|wine] [tree|nn|boost|svm|knn]
```
### Args:
1. Dataset either census or wine
2. Model
    - tree for Decision Tree
    - nn for Neural Network
    - boost for Boosted Decision Tree
    - SVM for Support Vector Machine
    - KNN for K-Nearest Neighbors
    
### Examples:

```bash
python run.py census tree # Census - Decision Tree
python run.py census nn # Census - Neural Network
python run.py wine knn # Wine - K nearest neighbors
# And so on...
```