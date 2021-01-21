from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver
from sacred import cli_option, Experiment

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from lib.utils import build_X_y

import pandas as pd
import importlib
import argparse 

if __name__== '__main__':
    parser = argparse.ArgumentParser(description = 'Run a simple model experiment.')
    parser.add_argument('--params')
    args = parser.parse_args()
    params = eval(args.params)
    
    mongo_user = 'alice'
    mongo_pass = 'bob'
    mongo_host = 'localhost'

    ex = Experiment(params['name'])

    mongo_url = 'mongodb://{0}:{1}@{2}:27017/' \
            'admin?authMechanism=SCRAM-SHA-1'.format(mongo_user, mongo_pass, mongo_host)

    #ex.observers.append(MongoObserver.create(url = mongo_url, db_name = 'admin'))
    ex.observers.append(FileStorageObserver('runs'))

    @ex.config 
    def cfg():
        experiment_params = params
        model_params = params['model']
        module = importlib.import_module(model_params['module'])
        model = getattr(module, model_params['model'])
        kwargs = model_params['kwargs']

    @ex.capture
    def get_model(model, **kwargs):
        return model(**kwargs)

    @ex.main
    def run():
        X, y = build_X_y(params)
        seed = params['model']['kwargs']['random_state']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = params['test_size'], random_state = seed)
        clf = get_model()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        pd.concat([X,y, pd.Series(clf.predict(X))], axis = 1).to_csv('./data/Xy.csv')
        _, pvalues = f_regression(X, y)

        results = {
            'rmse': mean_squared_error(y_pred, y_test, squared = False), 
            'pvalues': dict(zip(list(X.columns), pvalues))
        }

        if params['model']['module'] == 'sklearn.linear_model':
            results['coefficients'] = dict(zip(list(X.columns), clf.coef_))
            results['intercept'] = clf.intercept_
            results['r2'] = r2_score(y_test, y_pred) 

        return results
    
    ex.run()
