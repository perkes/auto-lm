
# AutoLM

A fully automatic suite to keep track of regression experiments.

### Prerequisites

- pip
- virtualenv


To get **virutalenv**: 
```
pip install virtualenv
``` 

### Installing

Create a new virtual environment:
```
virtualenv -p python3 venv
```    

Start the virtual environment:
```
source venv/bin/activate
```    
Install requirements with pip:
```
pip install -r requirements.txt
```

### Running

Once the installation phase is complete, you can run the experiments:
```
python run_experiments.py
```

This will run all experiments in the JSON file referenced by run_experiments.py, 
currently ./data/linear_experiments.json. There's a second json in the same
folder, non_linear_experiments.json. The experiments' results will be saved in
the runs folder (created after running the experiments), which will have one
folder for each experiment, each of these folders will have a json file called
run.json with the results. By default, these experiments are saved on a MongoDB
database, created by spinning up a Docker container. This release was modified
to write the results locally in the run folder to simplify the setup. The
MongoDB database can be used with Omniboard or Sacredboard, which are web views
of the experiments (dashboards actually).

## Built With

* [Sklearn](https://pypi.org/project/scikit-learn/) - Machine Learning Framework.
* [Pandas](https://pandas.pydata.org/) - Data manipulation.
* [Sacred](https://pypi.org/project/sacred/) - Saving experiment data.

## Authors

* **Jonathan Perkes** - [LinkedIn](https://www.linkedin.com/in/jonathan-perkes/)
