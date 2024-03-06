import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from GradientBoostingRegressor_Optimizer import Optimize as GBO
from GradientBoostingRegressor_Optimizer import GD_RandomSearchOutput
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import TransformerMixin,BaseEstimator

class SelectColumns( BaseEstimator, TransformerMixin ):
    # pass the function we want to apply to the column 'SalePrice'
    def __init__(self,columns):
        self.columns = columns
    def fit(self,xs,ys,**params):
        return self
    def transform(self,xs):
        return xs[self.columns]
    
if __name__ == "__main__":

    #load data and divide into train/test
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    ys_train = train['yield']
    xs_train = train.drop(columns=['yield','id','andrena','osmia','MaxOfUpperTRange','MinOfUpperTRange','AverageOfUpperTRange','MaxOfLowerTRange','MinOfLowerTRange','AverageOfLowerTRange','RainingDays','AverageRainingDays'])
    #xs_train = train.drop(columns=['yield','id'])

    # for col in train.columns:
    #     if col != 'yield':
    #         train.plot(x=col,y='yield',kind='scatter')
    #         plt.show()

    #let's see some transform graphs:



    grid = {
        #'regressor__min_samples_split':['squared_error','absolute_error','huber','quantile'],
        'regressor__learning_rate':[0.1,0.5],
        'regressor__n_estimators':[50,500],
        'regressor__subsample':[0.5,0.99],
        'regressor__min_samples_split':[2,20],
        'regressor__min_samples_leaf':[1,10],
        'regressor__min_weight_fraction_leaf':[0.0,0.1],
        'regressor__max_depth':[2,5],
        'regressor__min_impurity_decrease':[0.0,10.0]
        #'regressor__random_state':[1,50],
        #'regressor__max_features':['sqrt','log2'],
        #'regressor__max_leaf_nodes':[2,100]
    }

    grid_rand = {
        #'regressor__min_samples_split':['squared_error','absolute_error','huber','quantile'],
        'regressor__learning_rate':[i for i in np.arange(0.1,0.5,0.1)],
        'regressor__n_estimators':[i for i in range(50,501)],
        'regressor__subsample':[i for i in np.arange(0.5,0.99,0.1)],
        'regressor__min_samples_split':[i for i in range(2,21)],
        'regressor__min_samples_leaf':[i for i in range(1,11)],
        'regressor__min_weight_fraction_leaf':[i for i in np.arange(0.0,0.1,0.1)],
        'regressor__max_depth':[i for i in range(2,6)],
        'regressor__min_impurity_decrease':[i for i in np.arange(0.0,10.0,0.1)]
        #'regressor__random_state':[1,50],
        #'regressor__max_features':['sqrt','log2'],
        #'regressor__max_leaf_nodes':[2,100]
    }

    grid_fin = {'regressor__subsample': [0.7999999999999999], 	'regressor__n_estimators': [99], 	'regressor__min_weight_fraction_leaf': [0.0], 	'regressor__min_samples_split': [11], 'regressor__min_samples_leaf': 	[9], 'regressor__min_impurity_decrease': [1.2000000000000002], 	'regressor__max_depth': [4], 'regressor__learning_rate': [0.101]}

    steps=[
        ('column_select',SelectColumns(['clonesize','honeybee','bumbles','fruitset','fruitmass','seeds'])),
        ('scaler',MinMaxScaler()),
        #('PCA',PCA()),
        ('regressor', GradientBoostingRegressor(subsample=0.7999999999999999,n_estimators=99,min_weight_fraction_leaf=0.0,min_samples_split=11,min_samples_leaf=9,min_impurity_decrease=1.2000000000000002,max_depth=4,learning_rate=0.101))
    ]
    pipe = Pipeline( steps )
    pipe.fit(xs_train,ys_train)
    predictions = pipe.predict(test)
    output = pd.DataFrame({'id':test['id'],'yield':predictions})
    output.to_csv('output.csv',index=False)
    import random
    row_num = random.randint(40,500)
    sample_row = test.iloc[[row_num]]
    sample_row_p = test.iloc[row_num]
    print("___________EXAMPLE_OUTPUT___________")
    print(f"For an input instance where:\n\n{sample_row_p}")
    print(f"\n The predicted yield is: {pipe.predict(sample_row)[0]} lbs")
    #search = GridSearchCV(pipe, grid_fin, scoring='neg_mean_absolute_error',n_jobs=-1)
    #search.fit(xs_train,ys_train)
    #print(search.best_score_)





    """THE BELOW CODE WAS USED TO OPTIMIZE HYPERPARAMETERS"""
    # pipe = Pipeline( steps )
    # #print(grid_rand)
    # search = RandomizedSearchCV(pipe, grid_rand, scoring='neg_mean_absolute_error',n_iter=10)
    # search.fit(xs_train,ys_train)
    # print(f"Randomized search best score:{search.best_score_}")
    # print(f"Randomized search best params:{search.best_params_}")
    # GB = GD_RandomSearchOutput(search.best_score_,search.best_params_,steps, xs_train, ys_train)
    # GB.GD()
    # search = RandomizedSearchCV(pipe, grid_rand, scoring='neg_mean_absolute_error',n_iter=15)
    # search.fit(xs_train,ys_train)
    # print(f"Randomized search 15 iter best score:{search.best_score_}")
    # print(f"Randomized search 15 iter best params:{search.best_params_}")
    #Boosting_optimizer = GBO(xs_train, ys_train, steps, grid, 'neg_mean_absolute_error',400,2)
    #Boosting_optimizer.fit()
    # search = GridSearchCV(pipe, grid, scoring = 'neg_mean_absolute_error', n_jobs = -1)

    # search.fit(xs_train,ys_train)

    # print("Gradient Boosting Regressor:")
    # print("MAE: " + str(abs(search.best_score_)))
    # print("Best params: ")
    # print(search.best_params_)
