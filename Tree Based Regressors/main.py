import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

import numpy as np

class SelectColumns( BaseEstimator, TransformerMixin ):
    # pass the function we want to apply to the column 'SalePrice'
    def __init__(self,columns):
        self.columns = columns
    def fit(self,xs,ys,**params):
        return self
    def transform(self,xs):
        return xs[self.columns]


if __name__ == "__main__":
    #load data:
    data = pd.read_csv('AmesHousing.csv')
    xs = data.drop(columns=['SalePrice'])
    ys = data['SalePrice']

    for column in xs.columns: xs[column] = xs[column].fillna(0) 

    num_reg_columns = len(xs.columns)

    #numerical features we want to use:
    important_attributes = ['1st Flr SF','2nd Flr SF','BsmtFin SF 1','BsmtFin SF 2','Bsmt Unf SF','Overall Qual','Year Built','PID','Lot Area','Wood Deck SF','Lot Frontage','Year Remod/Add','Overall Cond','Mas Vnr Area','Bedroom AbvGr','Bsmt Full Bath','Full Bath','Kitchen AbvGr','TotRms AbvGrd','Fireplaces','Garage Yr Blt','Garage Cars','Garage Area','Open Porch SF','Enclosed Porch','3Ssn Porch','Screen Porch','Pool Area','Misc Val','Mo Sold','Yr Sold']
    
    #dummified features we want to use
    attributes_to_convert = ['Exter Qual','House Style','Kitchen Qual','Bsmt Qual','Bsmt Exposure','Garage Qual']
    converted_attribute_names = []
    #dummify these features:
    for attribute in attributes_to_convert:
        dummified = pd.get_dummies(xs[attribute],'dum')
        xs = pd.concat([xs,dummified],axis=1)
        converted_attribute_names.append(dummified.columns)

    num_of_elements = len(important_attributes)

    grid_of_elements = []
    index = 1

    #creates a grid with all combinations of important_attributes
    while(index<num_of_elements):
        for attribute_index in range(index):
            append_1 = important_attributes[attribute_index:index+attribute_index]
            grid_of_elements.append(append_1)
            for converted_attribute in converted_attribute_names:
                for i in range(len(converted_attribute)): append_1.append(converted_attribute[i])
                grid_of_elements.append(append_1)
        index = index + 1

    #remove duplicates:
    unique_entries = []
    for elem in grid_of_elements:
        if elem not in unique_entries:
            unique_entries.append(elem)
    grid_of_elements = unique_entries

    for column in xs.columns: xs[column] = xs[column].fillna(0)

#Starting with Linear Regression:
    grid = {
        'column_select__columns':grid_of_elements,

        'linear_regression':[
            LinearRegression(n_jobs=-1),
            TransformedTargetRegressor(
                LinearRegression(n_jobs=-1),
                func=np.sqrt,
                inverse_func=np.square),
            TransformedTargetRegressor(
                LinearRegression(n_jobs=-1),
                func=np.cbrt,
                inverse_func=lambda y: np.power(y,3)),
            TransformedTargetRegressor(
                LinearRegression(n_jobs=-1),
                func=np.log,
                inverse_func=np.exp),
        ]
    }

#create regressor:
    regressor = TransformedTargetRegressor(
        LinearRegression(n_jobs=-1),
        func = np.sqrt,
        inverse_func=np.square
    )

#create pipeline:
    steps=[
        ('column_select',SelectColumns(['Gr Liv Area'])),
        ('linear_regression',regressor),
    ]

    pipe=Pipeline( steps )
    
    search = GridSearchCV(pipe,grid,scoring='r2',n_jobs=-1)


    index = 0
    search.fit(xs,ys)

    print("Linear Regression:")
    print("R-squared: " + str(search.best_score_))
    print("Best params: ")
    print(search.best_params_)


    #Now on to the Random Forest Regressor. We'll use the same training set, as well as the same column grid.
    #create grid:
    all_features = []
    all_features = grid_of_elements[0]
    for i in grid_of_elements:
        if(len(i)>len(all_features)):
            all_features = i

    grid = {
        'column_select__columns':[all_features],
        'regressor__n_estimators':[100,400,500],
        'regressor__min_samples_split':[2,3,4]
    }

    steps=[
        ('column_select',SelectColumns(['Gr Liv Area'])),
        ('regressor', RandomForestRegressor(n_jobs = -1))
    ]

    pipe = Pipeline( steps )

    search = GridSearchCV(pipe, grid, scoring = 'r2', n_jobs = -1)

    search.fit(xs,ys)

    print("Random Forest Regressor:")
    print("R-squared: " + str(search.best_score_))
    print("Best params: ")
    print(search.best_params_)

    #Now on to the Decision Tree Regressor. We'll use the same training set, as well as the same column grid.
    #create grid:
    grid = {
        'column_select__columns':[all_features],
        'regressor__min_samples_split':[2,3,5,10,20,30,50],
        'regressor__splitter':["best","random"]
    }

    steps=[
        ('column_select',SelectColumns(['Gr Liv Area'])),
        ('regressor', DecisionTreeRegressor())
    ]

    pipe = Pipeline( steps )

    search = GridSearchCV(pipe, grid, scoring = 'r2', n_jobs = -1)

    search.fit(xs,ys)

    print("Decision Tree Regressor:")
    print("R-squared: " + str(search.best_score_))
    print("Best params: ")
    print(search.best_params_)

    #Now on to the Gradient Boosting Regressor. We'll use the same training set, as well as the same column grid.
    #create grid:
    grid = {
        'column_select__columns':[all_features],
        'regressor__min_samples_split':[2,5],
        'regressor__n_estimators':[200,1000],
        'regressor__min_samples_leaf':[2,3],
        'regressor__max_depth':[3,4]
    }

    steps=[
        ('column_select',SelectColumns(['Gr Liv Area'])),
        ('regressor', GradientBoostingRegressor())
    ]

    pipe = Pipeline( steps )

    search = GridSearchCV(pipe, grid, scoring = 'r2', n_jobs = -1)

    search.fit(xs,ys)

    print("Gradient Boosting Regressor:")
    print("R-squared: " + str(search.best_score_))
    print("Best params: ")
    print(search.best_params_)