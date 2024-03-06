import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.model_selection import train_test_split,GridSearchCV
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
    neighborhoods = pd.get_dummies(xs["Neighborhood"])

    for column in xs.columns: xs[column] = xs[column].fillna(0) 

    num_reg_columns = len(xs.columns)
    xs = pd.concat([xs,neighborhoods],axis=1)

    #convert neighborhoods to numerical (boolean) features and append to our "important_attributes"
    neighborhood_column_names = xs.iloc[:,num_reg_columns:len(xs.columns)-1]
    neighborhood_column_names = neighborhood_column_names.columns.values.tolist()

    important_attributes = ['1st Flr SF','2nd Flr SF','BsmtFin SF 1','BsmtFin SF 2','Bsmt Unf SF','Overall Qual','Year Built','PID','Lot Area','Wood Deck SF','Lot Frontage','Year Remod/Add','Overall Cond','Mas Vnr Area','Bedroom AbvGr','Bsmt Full Bath','Full Bath','Kitchen AbvGr','TotRms AbvGrd','Fireplaces','Garage Yr Blt','Garage Cars','Garage Area','Open Porch SF','Enclosed Porch','3Ssn Porch','Screen Porch','Pool Area','Misc Val','Mo Sold','Yr Sold','Neighborhood']

    num_of_elements = len(important_attributes)

    grid_of_elements = []
    index = 1

    #creates a grid with all combinations of important_attributes
    while(index<num_of_elements):
        for attribute_index in range(index):
            grid_of_elements.append(important_attributes[attribute_index:index+attribute_index])
        index = index + 1

    #add neighborhoods:
    for i in grid_of_elements:
        i.extend(neighborhood_column_names)


#create grid:
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

    neighborhoods = []
    index = 0

    #tested this way to change neighborhoods to numbers (not actually using at atm):
    for index,row in enumerate(xs["Neighborhood"]):
        if(row in neighborhoods):
            xs.loc[index,'Neighborhood'] = neighborhoods.index(row)
        else:
            neighborhoods.append(row)
            xs.loc[index,'Neighborhood'] = neighborhoods.index(row)
    print(xs)
    try:
        search.fit(xs,ys)
    except:
        None
    print(search.best_score_)
    print(search.best_params_)