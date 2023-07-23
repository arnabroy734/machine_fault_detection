import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


class Clustering:
    """
    1. Read the preprocessed data
    2. Take the rolling average
    3. Drop the nan values
    4. Split the data in train and test
    6. Train the model
    7. Save the test score - precision and recall
    """
    def __init__(self, rolling_window, method, test_ratio):
        """
        Parameters:
        rolling_window(int) - no. of lags to smoothen the data
        method(str) - "sma" for simple moving average and "ewm" for exponential moving average
        test_ratio(float) - percentage of test data
        """
        self.rolling_window = rolling_window
        self.method = method
        self.test_ratio = test_ratio
        
    
    def fit(self, data):
        """
        Parameter:
        data - the pd dataframe of the training data
        """
        y = data['status']
        X = data.drop(['status'], axis=1)
        
        if self.method == 'sma':
            X = X.rolling(window=self.rolling_window).mean()
        elif self.method == 'ewm':
            X = X.ewm(com=self.rolling_window).mean()
        
        # Drop na vallues from X and corresponding indices of y also
        not_na_idx = X[X.iloc[:,0].notna()].index
        X = X.dropna()
        y = y[not_na_idx]
        

        # Train test split
        total_size = X.shape[0]
        test_size = int(total_size*self.test_ratio)
        X_train, X_test = X.iloc[0:total_size-test_size],X.iloc[total_size-test_size:]
        y_train, y_test = y[0:total_size-test_size],y[total_size-test_size:]
        
        # Scale the data and fit K Means Clustering
        self.scaler = StandardScaler() 
        X_scaled = self.scaler.fit_transform(X_train)
        self.cluster = KMeans(n_clusters=2, n_init='auto' ).fit(X_scaled)
        
        # Find which class belongs to which cluster
        no_of_1 = np.count_nonzero(self.cluster.labels_)
        no_of_0 = self.cluster.labels_.shape[0] - no_of_1
            
        if no_of_1 > no_of_0:
            self.invert_label = True
        else:
            self.invert_label = False
            
        y_test_pred = self.cluster.predict(self.scaler.transform(X_test))
            
        if self.invert_label:
            y_test_pred = 1-y_test_pred
            self.cluster.labels_ = 1 - self.cluster.labels_
            
        # Saving the scoring 
        self.train_recall = recall_score(y_train, self.cluster.labels_)
        self.test_recall  = recall_score(y_test, y_test_pred)
        self.train_precision = precision_score(y_train, self.cluster.labels_)
        self.test_precision = precision_score(y_test, y_test_pred)
        self.test_f1 = f1_score(y_test, y_test_pred)
        self.train_f1 = f1_score(y_train, self.cluster.labels_)
        
    def predict(self, X):
        """
        Parameter: X is a pd data frame having only the features
        Description:
        1. It will calculate the rolling average from rolling window
        2. Then scaling will be done
        3. Then it will find the predicted class level (1 - Fault, 0 - Healthy) of the last data point
        4. In case it returns -1 means data frame size is less than the window
        """
        if X.shape[0] < self.rolling_window:
            return -1
        
        # Moving average and drop nan
        if self.method == 'sma':
            X = X.rolling(window=self.rolling_window).mean()
        elif self.method == 'ewm':
            X = X.ewm(com=self.rolling_window).mean()
        X = X.dropna()
        
        query_pt = X.iloc[-1].values.reshape(1,-1) # Last point
        query_pt = self.scaler.transform(query_pt)
        prediction = self.cluster.predict(query_pt)[0]
        
        if self.invert_label:
            prediction = 1 - prediction
        
        return prediction
        