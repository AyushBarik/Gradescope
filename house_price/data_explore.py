import pandas as pd
import numpy as np


train_path = "data/train.csv"

df = pd.read_csv(train_path)

# Number of rows in data
print(f"Number of rows: {len(df)}")

# Columns
print(f"Number of columns: {len(df.columns)}")


"""
A. Preparing the data
    Step 1: remove all the non numerical columns
    Step 2: Convert the dataframe to numpy

B. Training the model
    Step 1: Create X and Ys
    Step 2: Create coefficients (a,b,c) assume linear equation
    Step 3: Assume some random coefficient initialize
    Step 4: Compute the current Loss/Error
    Step 5: Iterate and update coefficients
        5.a Formula for gradient
        5.b update the coefficient
        
"""

label_column = "SalePrice"

# A.1
features = df.drop(columns=[label_column])
labels = df[label_column]
numeric_features = features.select_dtypes(include=['number'])

# A.2 
numeric_features_array = numeric_features.values # [N, 37]
labels_array = labels.values #[N]

# print(numeric_features_array.shape, labels_array.shape)

# Training a model
# B

X = numeric_features_array
Y = labels_array
# print(Y.shape) 1460
N = X.shape[0]
# print(X)
# # print(X.shape) 1460,37
# print(Y)
guess = np.zeros(shape=[37, 1])
# guess
# ax + b
# a 
prediction = X @ guess
print(Y)  
print(prediction)  

max_iteration = 1
for i in range(len(guess)):
    
    # stepD = 0.01*
    for step in range(max_iteration):
        prediction = X @ guess # [N, 37] @ [37,1] = [N, 1]
        # print(prediction.shape)
        # print(prediction)
        loss = np.square(prediction - Y)  # 1460,1  1460 
        
        print(loss.shape)
        print(prediction.shape)
        print(Y.shape)
        print((prediction - Y).shape)
        
        Hcopy = np.copy(guess)
        Lcopy = np.copy(guess)
        Hcopy[i,:] += 0.1
        Lcopy[i,:] -= 0.1
        Hloss = np.square((X @ (Hcopy)) - Y)
        Lloss = np.square((X @ (Lcopy)) - Y)
        # print(Hloss)
        # print(Lloss)
        
        
        # print(np.sum(Hloss))
        # print(np.sum(Lloss))
        if(np.sum(Hloss) > np.sum(Lloss)): 
            guess[i,:] -= 0.1
            
        if(np.sum(Hloss) < np.sum(Lloss)):
            guess[i,:] += 0.1
        

# print(guess[i,:])

            
    

