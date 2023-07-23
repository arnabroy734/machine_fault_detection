from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest


class Isolation:
    """
    1. Take the rolling average
    2. Drop the nan values
    4. Split the data in train and test
    6. Train the model
    7. Save the test score - precision and recall
    """
    def __init__(self, rolling_window, method, test_ratio, contamination, max_features):
        """
        Parameters:
        rolling_window(int) - no. of lags to smoothen the data
        method(str) - "sma" for simple moving average and "ewm" for exponential moving average
        test_ratio(float) - percentage of test data
        """
        self.rolling_window = rolling_window
        self.method = method
        self.test_ratio = test_ratio
        self.contamination = contamination
        self.max_features = max_features
        
    
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
        
        # Fit Isolation Forest on this data
        self.model = IsolationForest(contamination=self.contamination, max_features=self.max_features)
        self.model.fit(X_train)
        
        # Find the prediction
        y_test_pred = self.model.predict(X_test)
        y_train_pred = self.model.predict(X_train)
        
        y_train_pred[y_train_pred == 1] = 0
        y_train_pred[y_train_pred == -1] = 1
        
        y_test_pred[y_test_pred == 1] = 0
        y_test_pred[y_test_pred == -1] = 1
            
        # Saving the scoring 
        self.train_recall = recall_score(y_train, y_train_pred)
        self.test_recall  = recall_score(y_test, y_test_pred)
        self.train_precision = precision_score(y_train, y_train_pred)
        self.test_precision = precision_score(y_test, y_test_pred)
        self.test_f1 = f1_score(y_test, y_test_pred)
        self.train_f1 = f1_score(y_train, y_train_pred)
        
    def predict(self, X):
        """
        Parameter: X is a pd data frame having only the features
        Description:
        1. It will calculate the rolling average from rolling window
        2. Then it will find the predicted class level (1 - Fault, 0 - Healthy) of the last data point
        3. In case it returns -1 means data frame size is less than the window
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
        prediction = self.model.predict(query_pt)
        
        if prediction == -1:
            return 1 # Outlier pt - faulty data
        else:
            return 0 # Healthy data