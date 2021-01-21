from sklearn.preprocessing import PolynomialFeatures
from functools import reduce
import pandas as pd
import numpy as np

eps = 0.0001

def is_dummy(dummy, dummies):
    return '_'.join(dummy.split('_')[:-1]) in dummies

def include_polynomial(feature_name, polynomial_features):
    poly = feature_name.split()
    if len(poly) == 1:
        return True
    
    feature = ['_'.join(feature.split('_')[:-1]) for feature in poly]
    
    return feature in polynomial_features or feature[::-1] in polynomial_features

def exclude_feature(feature_name, features_to_drop):
    return feature_name in features_to_drop or (len(feature_name.split()) == 1 and feature_name.split('_')[0] in features_to_drop)

def build_X_y(params):
    for input_ in params['inputs']:
        df = pd.read_csv(input_['filename'])
        
        if 'filter' in input_:
            for f in input_['filter']:
                df = df[getattr(df[f['by']], '__' + f['op'] + '__')(f['val'])]

        df = df[input_['include']]

        if 'dummies' in input_:
            df[input_['dummies']] = df[input_['dummies']].astype(str)
            dummies = pd.get_dummies(df[input_['dummies']])
            non_dummies = df.drop(input_['dummies'], axis = 1)
            df = pd.concat([dummies, non_dummies], axis = 1)

        if 'split' in input_:
            for criteria in input_['split']:
                values = df[criteria['by']].unique()
                for val in values:
                    df.loc[df[criteria['by']] == val, criteria['metric'] + '_' + val.lower()] = df[criteria['metric']]
                    df.loc[df[criteria['by']] != val, criteria['metric'] + '_' + val.lower()] = 0
                df.drop(criteria['metric'], axis = 1, inplace = True)

        input_['df'] = df.groupby(input_['groupby']).sum().reset_index()
        
        if 'dummies' in input_:
            dummies = [feature for feature in list(df.columns) if is_dummy(feature, input_['dummies'])]

    df = reduce(lambda left, right: pd.merge(left['df'], right['df'], left_on = left['join_on'], right_on = right['join_on'], how = 'left'), params['inputs'])
    
    df[params['target']] = df[params['target']].shift(params['shift_target_by'])
    df = df[df[params['target']].notna()]
    
    if 'transform' in params:
        for tf in params['transform']:
            if tf['function'] == 'sigmoid':
                df[tf['name']] = df[tf['metric']].transform(lambda x: 1/(1 + np.exp(-1*tf['v']*x)))
            if tf['function'] == 'log':
                df[tf['name']] = df[tf['metric']].transform(lambda x: np.log(max(x,eps)))
    
    if 'shift' in params:
        for lag in params['shift']:
            start = lag['from']
            end = lag['to']
            while start < end:
                df[lag['name'] + str(start)] = df[lag['metric']].shift(start)
                start += 1
            df = df[df[lag['name'] + str(end-1)].notna()]
            
    if 'sma' in params:
        for roll in params['sma']:
            df[roll['name']] = df[roll['metric']].rolling(roll['by']).mean()
            
    if 'ewm' in params:
        for ewm in params['ewm']:
            df[ewm['name']] = df[ewm['metric']].ewm(halflife = ewm['halflife'], adjust = False).mean()
    
    X, y = df.drop(params['target'], axis = 1), df[params['target']]
    X.fillna(0, inplace = True)
    
    if dummies:
        X[dummies] = X[dummies] > 0
        X[dummies] = X[dummies].astype('int64')
    
    if 'polynomial_features' in params:
        poly = PolynomialFeatures(2, interaction_only = True, include_bias = False)
        X_poly = poly.fit_transform(X)
        X = pd.DataFrame(X_poly, columns = poly.get_feature_names(X.columns))
        X = X[[col for col in X.columns if include_polynomial(col, params['polynomial_features'])]]
        
        if 'drop_after_poly' in params:
            drop = params['drop_after_poly']
            X = X[[col for col in X.columns if not exclude_feature(col, drop)]]
    
    if 'drop' in params:
        X.drop(params['drop'], axis = 1, inplace = True)
        
    return X, y