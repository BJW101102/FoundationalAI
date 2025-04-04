import math
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split


def load_mpg_dataset():
    auto_mpg = fetch_ucirepo(id=9)

    X = auto_mpg.data.features
    y = auto_mpg.data.targets

    data = pd.concat([X, y], axis=1)

    cleaned_data = data.dropna()

    X = cleaned_data.iloc[:, :-1]
    y = cleaned_data.iloc[:, -1]


    # Do a 70/30 split (e.g., 70% train, 30% other)
    X_train, X_leftover, y_train, y_leftover = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,    
        shuffle=True,      
    )

    # Split the remaining 30% into validation/testing (15%/15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_leftover, y_leftover,
        test_size=0.5,
        random_state=42,
        shuffle=True,
    )

    # Compute statistics for X (features)
    X_mean = X_train.mean(axis=0) 
    X_std = X_train.std(axis=0)   
    
    # Standardize X
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    # Compute statistics for y (targets)
    y_mean = y_train.mean()  
    y_std = y_train.std()   

    # Standardize y
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    X_train = np.array(X_train)
    y_train = np.array(y_train).reshape(-1, 1)
    X_val   = np.array(X_val)
    y_val   = np.array(y_val).reshape(-1, 1)
    X_test  = np.array(X_test)
    y_test  = np.array(y_test).reshape(-1, 1)


    return (X_train, y_train), (X_val, y_val), (X_test, y_test) 