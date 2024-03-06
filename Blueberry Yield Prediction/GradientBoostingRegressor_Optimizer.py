#The goal: pass in the data, this will automatically optimize ALL of gradient boosting attributes with gradient descent and/or simulated annealing.
#make it a wrapper for GridSearchCV, so you pass it pipe, steps, etc... but then you fuck with it
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import copy, random, math
"""THIS CLASS IS G.D. IN COMBINATION WITH RANDOM SELECTION.  IT WAS NOT USED FOR THE FINAL SOLUTION"""
class Optimize:
    def __init__(self, xs, ys, steps, grid, scoring, min_initial_value, optimize_stages):#grid with upper and lower bounds
        self.xs = xs
        self.ys = ys
        self.steps = steps
        self.grid = grid
        self.scoring = scoring
        self.min_initial_value = min_initial_value
        self.optimize_stages = optimize_stages

    def fit(self):
        total_iterations = 0
        # regressor_step = self.steps[len(self.steps)-1]
        # regressor = regressor_step[1]
        pipe = Pipeline(self.steps)
        #run GD 10 times at 10 evenly spaced values along the feature space:
        intervals = 2
        overall_best_score = float('inf')
        overall_best_grid = self.grid
        for n in range(1,intervals):
            temp_grid = copy.deepcopy(self.grid)
            #print(f"Original grid:{temp_grid}")
            first_iteration = True
            #We need to assign the whole parameter grid a random start state:
            for tempkey, tempvalue in temp_grid.items():
                #assign a random value to each element:
                is_float = isinstance(tempvalue[0],float)
                if is_float: new_x = random.uniform(tempvalue[0],tempvalue[1])#n*(tempvalue[1]-tempvalue[0])/intervals
                else: new_x = random.randint(tempvalue[0],tempvalue[1])#round(n*(tempvalue[1]-tempvalue[0])/intervals)#random.randint(tempvalue[0],tempvalue[1])
                temp_grid[tempkey] = [new_x]
            #print(f"Random state grid:{temp_grid}")
            count = 0
            for key, value in self.grid.items():
                count = count+1
                if count>self.optimize_stages:break
                print(f"\nOptimizing {key} with value {value}")
                #print(f"GRID: {temp_grid}")
                bottom_limit = value[0]
                top_limit = value[1]
                new_x = temp_grid[key][0]
                #figure out if the number needs to remain an int or a float:
                is_float = isinstance(bottom_limit,float)

                #print(f"Bottom limit {bottom_limit} top limit {top_limit}")
                
                # if is_float:
                #     x = temp_grid[key][0]+0.25*temp_grid[key][0]
                #     if x>top_limit: temp_grid[key][0]-0.25*x
                # else:
                #     x = temp_grid[key][0]+0.5*temp_grid[key][0]
                #     if x>top_limit: temp_grid[key][0]-0.5*temp_grid[key][0]
                x = (bottom_limit+top_limit)/2 #first val is middle
                if not is_float: x = round(x)
                alpha = 1
                score = float('inf')

                #get an initial reference point score (this is valid for ALL parameters!):
                if first_iteration:
                    rand_save = temp_grid[key][0] #save this value
                    temp_grid[key] = [x]
                    search = GridSearchCV(pipe,temp_grid,scoring=self.scoring,n_jobs=-1)
                    total_iterations = total_iterations + 1
                    search.fit(self.xs,self.ys)
                    best_score = abs(search.best_score_)
                    temp_grid[key] = [rand_save] #switch back to original rand number
                    first_iteration = False
                if best_score > self.min_initial_value or best_score>overall_best_score*0.01+overall_best_score: break
                print(f"Testing value: {x} against {new_x}")
                last_new_x = new_x
                iter_num = 0
                # print("\n GRID:")
                # print(temp_grid)
                while alpha > 1/pow(2,5):
                    #We're starting with a reference score, as well as a ready value of x that is less than prev,
                    #so we can run the first iteration right away:
                    print(f"a: {alpha}, Best score:{abs(best_score)}, x: {x}")
                    search = GridSearchCV(pipe,temp_grid,scoring=self.scoring,n_jobs=-1)
                    total_iterations = total_iterations + 1
                    search.fit(self.xs,self.ys)
                    score = abs(search.best_score_)
                    
                    print(f"Initial scores: {best_score} against {score} with new_x:{new_x}")
                    #Find the derivative of score with respect to our parameter and calculate new x:
                    if is_float:
                        if x != new_x: dsdx=(score-best_score)/(x-new_x)
                    else:
                        if x != new_x:
                            dsdx=math.ceil(abs((score-best_score)/(x-new_x)))*((score-best_score)/(x-new_x))/abs((score-best_score)/(x-new_x))
                    print(f"dsdx = {dsdx}, a*dsdx = {alpha*dsdx}")
                    new_x = x - alpha*dsdx
                    if new_x == temp_grid[key]: break
                    if new_x < bottom_limit or new_x > top_limit:
                        #print(f"Skipping iteration with new_x:{new_x}, x:{x}")
                        alpha = alpha/2
                        continue
                    if not is_float:
                        new_x = round(new_x)
                    if abs(alpha*dsdx)<0.001: break
                    #last_new_x = new_x
                    temp_grid[key] = [new_x]
                    
                    #Check if we got a better result, if so assign to x:
                    if score<best_score: 
                        x = new_x
                        best_score = score
                    else:
                        alpha = alpha/2
                        #print(f"Skipping iteration with new_x:{new_x}, x:{x},score:{score},best_score:{best_score}")
                        continue
                    
                    iter_num = iter_num+1
                    if iter_num > 100: break
                temp_grid[key] = [x]
                print(f"Overall best value for {key}: {x} with score of: {best_score}")

            if best_score < overall_best_score and best_score<self.min_initial_value:
                print(f"Best score:{best_score}")
                print(f"Best params:{temp_grid}")
                overall_best_score = best_score
                overall_best_grid = temp_grid
        print(f"Overall best score: {overall_best_score}\n Best params:{overall_best_grid} \n Training iterations:{total_iterations}")

"""THIS CLASS USES GridSearchCV OUTPUT FOR G.D.  IT WAS USED FOR THE SOLUTION"""
class GD_RandomSearchOutput:
    def __init__(self, score_to_beat, params, steps, xs, ys):
        self.score_to_beat = score_to_beat
        self.params = params
        self.steps = steps
        self.xs = xs
        self.ys = ys
    def GD(self):
        keys_to_optimize = ['regressor__learning_rate','regressor__n_estimators']
        iterations = 1
        op_params = copy.deepcopy(self.params)
        pipe = Pipeline(self.steps)
        for key, value in self.params.items():
            op_params[key] = [self.params[key]]

        overall_best_score = self.score_to_beat
        overall_best_params = copy.deepcopy(op_params)

        for key, value in op_params.items():
            if key not in keys_to_optimize: continue
            is_float = isinstance(value[0],float)
            if is_float:
                op_params[key] = [op_params[key][0] + 0.01*op_params[key][0]]
            else:
                op_params[key] = [op_params[key][0] + 1]
            search = GridSearchCV(pipe, op_params, scoring = 'neg_mean_absolute_error', n_jobs = -1)
            search.fit(self.xs,self.ys)
            print(f"Optimizing:{key}")
            multiplier = 0.005
            for i in range(2):
                print(f"Best score:{overall_best_score}, last score: {search.best_score_}")
                if search.best_score_ > overall_best_score:
                    overall_best_score = search.best_score_
                    overall_best_params = copy.deepcopy(op_params)
                    try:
                        if is_float: op_params[key] = [op_params[key][0] + multiplier*op_params[key][0]]
                        else: op_params[key] = [op_params[key][0] + 1]
                        search = GridSearchCV(pipe, op_params, scoring = 'neg_mean_absolute_error', n_jobs = -1)
                        search.fit(self.xs,self.ys)
                    except:
                        if is_float: op_params[key] = [op_params[key][0] - multiplier*op_params[key][0]]
                        else: op_params[key] = [op_params[key][0] - 1]
                        break

                else:
                    try:
                        if is_float: op_params[key] = [op_params[key][0] - multiplier*op_params[key][0]]
                        else: op_params[key] = [op_params[key][0] - 1]
                        search = GridSearchCV(pipe, op_params, scoring = 'neg_mean_absolute_error', n_jobs = -1)
                        search.fit(self.xs,self.ys)
                    except:
                        if is_float: op_params[key] = [op_params[key][0] + multiplier*op_params[key][0]]
                        else: op_params[key] = [op_params[key][0] + 1]
                        break
                iterations = iterations+1
                multiplier = multiplier/2
        print(f"Overall best score:{overall_best_score} vs old best score:{self.score_to_beat}")
        print(f"Overall best params:{overall_best_params}")
        print(f"Iteration number:{iterations}")