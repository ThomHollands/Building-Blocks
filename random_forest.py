# implementing random forests from scratch
# doesnt work particularly well but at the moment cba to fix
import numpy as np
from random import randrange, seed
import pandas as pd
from csv import reader


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


def get_split(dataset, n_features):
    """ This function works out the best split along a variable axis to minimise the gini index of the data
    Gini index tells you about the purity of subclasses. It is minimised if all classes are the same.
   It picks n_features from which to evaluate. (doing all is v expensive)."""
    class_values = list(set(row[-1] for row in dataset)) # gets the class values if not specified separately
    b_index, b_value, b_score, b_groups = 999, 999, 999, None # best index, best value, best score, besst groups
    features = [] # initialise features to split on
    while len(features) < n_features:
        index = randrange(len(dataset[0])-1) # picks a random feature from the dataset
        if index not in features: # makes sure it hasnt already been picked
            features.append(index)
    for index in features:
        for row in dataset:
            groups = test_split(index,row[index],dataset)
            gini = gini_index(groups,class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

def test_split(index,value,dataset):
    ''' Splits a data set based on an index and an attribute value'''
    left,right = [],[]
    for row in dataset:
        if row[index]<value:
            left.append(row)
        else:
            right.append(row)
    return left,right

def gini_index(groups,y_values):
    ''' Calcs gini index for a split dataset'''
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0 #initialise
    # sum weighted gini index for each group
    for group in groups:
        size = float(len(group))
        if size == 0: # avoid divide by zero
            continue
        score = 0
        # score the group based on the score for each class
        for class_val in y_values:
            p = [row[-1] for row in group].count(class_val) / size # the predicted value
            score = score + p*p # square it for the gini index
        # weight the group score by its relative size
        gini = gini + (1.0 - score)*(size/n_instances)
    return gini

def to_terminal(group):
    ''' Creates a terminal predict value. Based on the class with the biggest frequency (modal)'''
    outcomes = [row[-1] for row in group]
    return max(set(outcomes),key=outcomes.count)

def split(node,max_depth,min_size,n_features,depth):
    ''' Creates child splits for a node, or make the node terminal and output value'''
    left,right = node['groups']
    del node['groups']
    # check for no split -- then terminate the node totally
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left+right) # both left and right are equal to prediction
    # check for max depth
    if depth >= max_depth: # forgot to terminate the node
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # if no left, then terminate the left side
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    # otherwise, split the left node
    else:
        node['left'] = get_split_dec(left)
        split(node['left'], max_depth, min_size, n_features,depth+1) # the recursive part of the function
    # if no right, then terminate the right side    
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split_dec(right)
        split(node['right'],max_depth,min_size,n_features,depth+1)
       
def build_tree(train,max_depth,min_size):
    root = get_split_dec(train) # splits the training data into two groups, gets root node
    split(root,max_depth,min_size,n_features,depth=1)
    return root

# make a prediction with a decision tree
def predict(node,row):
    if row[node['index']]< node['value']: # if value of node at given index is less than the splitting point (pick left side)
        if isinstance(node['left'],dict): # check that there are more nodes, that this isnt a terminus
            return predict(node['left'],row) # if it isn't a terminus, recurse
        else:
            return node['left']
    else: # if value of node at given index is greater (then we go into right branch)
        if isinstance(node['right'],dict):
            return predict(node['right'],row)
        else:
            return node['right']
def subsample(dataset,ratio):
    # ratio tells us what % of dataset to sample
    sample = []
    n_sample = round(len(dataset)*ratio) # number of samples
    while len(sample) < n_sample:
        # pick from dataset
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

def bagging_predict(trees,row):
    predictions = [predict(tree,row) for tree in trees]
    return max(set(predictions),key=predictions.count) # return the modal value of all the trees

def random_forest(train,test,max_depth,min_size,sample_size,n_trees,n_features):
    trees = []
    for i in range(0,n_trees):
        print(sample_size)
        sample = subsample(train,sample_size) # subsample some values
        tree = build_tree(train,max_depth,min_size) # build the tree for each subsample
        trees.append(tree) 
    predictions = [bagging_predict(trees,row) for row in test] # lets each tree vote
    return predictions

def csv_reader(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset



# Test the random forest algorithm
seed(2)
# load and prepare data
filename = 'sonar.all-data'
data = pd.read_csv(filename)
dataset = csv_reader(filename)
# convert string attributes to integers
for i in range(0, len(dataset[0])-1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 5
max_depth = 3
min_size = 1
sample_size = 1.0
n_features = int(np.sqrt(len(dataset[0])-1))
split_ind = int(np.floor(len(dataset)*0.7))
train = dataset[:split_ind]
test = dataset[split_ind:]
n_trees = 5
preds = random_forest(train,test,max_depth,min_size,sample_size,n_trees,n_features)
x = []
for i in range(len(test)):
    x.append(test[i][-1])
print(np.array(x)-np.array(preds))