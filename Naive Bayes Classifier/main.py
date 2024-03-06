import math as m
import pandas as pd

def log_mult(x,y):
    #multiplies two numbers 'x' and 'y' using logarithm properties to avoid underflow
    if((x == 0) or (y==0)):
        return 0
    else:
        return pow(2,m.log2(x)+m.log2(y))

def calculate_class_probs(class_type):
    #calculates the overall probability of each class occuring (number of this class)/(total classes)
    num_of_classes = len(class_type)
    class_probabilities = []
    classes = []

    [classes.append(i) for i in class_type]
    [class_probabilities.append(classes.count(i)/num_of_classes) for i in range(1,8)]
    return class_probabilities

def get_num_of_classes(class_type):
    #returns the number of each class in the dataset, e.g. 4 of class '1', 6 of class '2', etc...
    classes = []

    [classes.append(i) for i in class_type]
    return [classes.count(i) for i in range(1,8)]

def calculate_feature_prob(instance,training_data,num_of_each_class):
    '''
    Returns the probability score (NB score) for each feature to be in each class based on the training data.
    For example, given the feature "feathers", it may return:
    [0,0.04,0.3,0.1,0.0074,0.1231,0]
    Where the first element is the score for class '1', the second element is the score for class '2', etc...

    This function will do so for each provided feature, thus returning an array of arrays.
    So for an 'instance' (row) it will return an array of 7 elements for each feature, except "animal_name" and "class"
    '''
    alpha = 0.01
    instance_probs = [] #array with probabilities for each feature for each class

    feature_num = 1 #start at "hair", not "animal_name"
    while feature_num<17:#for each feature we have been given except the actual class
        current_feature = instance[feature_num]
        class_occurences = [0,0,0,0,0,0,0] #keep track of how many times this feature occurs in each class
        for index,row in training_data.iterrows():#find all occurences of current_feature being true and iterate class index
            if(row[feature_num]==current_feature):
                current_class = row['class_type']
                class_occurences[current_class-1]=class_occurences[current_class-1]+1

        #calculate probabilities for each:
        feature_num = feature_num + 1
        out = []

        #now calculate the NB score for each feature for each class, and do laplace smoothing:
        for number_of_this_class_in_dataset,feature_occurences_in_this_class in zip(num_of_each_class,class_occurences):
            out.append((feature_occurences_in_this_class+alpha)/(number_of_this_class_in_dataset+alpha*7)) #alpha laplace smoothing
        
        instance_probs.append(out)
    
    return instance_probs

def run_sample():
    data = pd.read_csv('zoo.csv') #read in data
    train = data.sample(frac = 0.7) #70% is training data
    test = data.drop(train.index)

    #let's first calculate the overall probability of each animal class:
    class_type = train['class_type']
    class_probabilities=calculate_class_probs(class_type)
    num_of_each_class = get_num_of_classes(class_type)

    #now lets iterate through each row in the test set and calculate probs:
    winning_class = []
    actual_class = []
    is_correct = []
    probabilities = []
    for row,instance in test.iterrows():
        #row is instance
        #now get probs for each instance:
        instance_probs = calculate_feature_prob(instance,train,num_of_each_class)
        #we now have the probability that each feature occurs for each class
        
        #Now multiply all the class scores together to get the scores for each class:
        scores = []
        for c in range(0,7):
            overall_class_prob = class_probabilities[c]
            score = 1
            chances_for_class = [feature_probs[c] for feature_probs in instance_probs]#feature probs is an array of probs for each class
            for chance in chances_for_class:
                score = log_mult(score,chance)

            score = score*overall_class_prob
            scores.append(score)
        
        #now that we have the scores, we need to normalise each one:
        sum_scores = sum(scores)
        probability_scores = [(i)/(sum_scores) for i in scores] #normalize to actual probabilities
        winning_score_index = probability_scores.index(max(probability_scores))
        winning_score = winning_score_index + 1 #add one, since zero-indexed
        probabilities.append(probability_scores[winning_score_index])
        winning_class.append(winning_score)

        #Get the actual class and make a new column for whether it was right or wrong:
        actual_class.append(instance['class_type'])
        if(winning_score==instance['class_type']):
            is_correct.append("CORRECT")
        else:
            is_correct.append("wrong")

    #get number of elements in common between winning and actual:
    total_guesses = len(winning_class)
    correct_guesses = len([winning_class[i] for i in range(total_guesses) if winning_class[i]==actual_class[i]])

    #calculate percentage we got correct:
    correct_percentage = correct_guesses/total_guesses*100

    #Format and print output
    test['predicted'] = winning_class
    test['probability'] = probabilities
    test['correct'] = is_correct
    print(','.join(test.columns))
    for index,row in test.iterrows():
        row_as_csv = ','.join(map(str,row))
        print(row_as_csv)
    return correct_percentage

if __name__ == "__main__":

    '''__Single run__'''
    result = run_sample()

    '''__Loop to test classifier over 50 runs__'''
    # a = []
    # for i in range(50):
    #     result = run_sample()
    #     a.append(result)
    #     print("Last score: " + str(result) + "%.  Average: " + str(sum(a)/len(a)) + "%")
    