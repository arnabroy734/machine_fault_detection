import stumpy
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.metrics import confusion_matrix
from log.logging import LOGGER
import logging

class MatrixProfiling:

    def __init__(self):
        self.profile_cutoffs = {}
        self.window = 130
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.faulty = None
        

    def find_distance_profile(self,fault_query, fault_t, start_id, end_id, column):
        """
        This is the job method of ProcessPool of find_cutoffs

        Description:
        1. Take slice of data from start index to end index
        2. Query one set data of window size=130 on other set and save the least distance
        3. Store least distances in a list and return the list and column name
        """
        try:
            distances = []
            for i in range(start_id, end_id - self.window):
                matches = stumpy.match(Q=fault_query[i:i+self.window][column], T=fault_t[column])
                distance = matches[0][0]
                distances.append(distance)
            return (distances, column)
        except Exception as e:
            return ([], None)
        
    def callback_update_profiles(self,future):
        """
        Description:
        Callback function of ThreadPool of find_cutoffs
        """
        distance, column = future.result()
        # print(len(distance), column)
        if column is not None:
            self.profile_cutoffs[column].extend(distance)

    def find_cutoffs(self, data):
        """
        1. Separate faulty data and save fault data for future
        2. Split the faulty data in two part of 50% split size
        3. Query one set data of window size=130 on other set and save the least distance
        4. Calculate the distances for each feature
        5. Take upper fence value (q3 + 1.5*IQR) as cutoff
        """
        try:

            faulty = data[data['status']==1]
            fault_query = faulty[0:int(faulty.shape[0]*0.5)]
            fault_t = faulty[int(faulty.shape[0]*0.5) : ]

            self.faulty = fault_t.drop(['status'], axis=1).copy() # Save data for use in prediction

            for column in data.columns:
                if column != 'status':
                    self.profile_cutoffs[column] = []


            with ProcessPoolExecutor() as executor:
                query_N = fault_query.shape[0]
                chunk = 2000
                start_idx = 0
                end_idx = chunk

                while True:
                    if end_idx > query_N:
                        end_idx = query_N
                    if start_idx == end_idx:
                        break

                    for column in self.profile_cutoffs.keys():
                        job = executor.submit(self.find_distance_profile, fault_query.copy(), fault_t.copy(), start_idx, end_idx, column)
                        job.add_done_callback(self.callback_update_profiles)
                    start_idx, end_idx = end_idx, end_idx + chunk
            
                executor.shutdown()

        
            for column in self.profile_cutoffs.keys():
                q1 = np.quantile(self.profile_cutoffs[column], 0.25)
                q3 = np.quantile(self.profile_cutoffs[column], 0.75)
                IQR = q3 - q1
                upper_fence = q3 + 1.5*IQR
                self.profile_cutoffs[column] = q3
            
            LOGGER.log_profiling(f"Successfully done profiling and found cut offs for features", logging.INFO)

        except Exception as e:
            LOGGER.log_profiling(f"Error in finding profile cutoffs - {e}", logging.ERROR)


    def predict(self, X):
        """
        Description:
        1. If X has size less than self.window return prediction = -1
        2. Slice from last of the dataframe having frame size self.window
        3. Find the least distance of this slice from self.faulty for all columns/features
        4. For any feature if score < cutoff the feature is anomalous/faulty
        5. If no. of faulty features > 3:
                prediction = 1
            else:
                prediction  = 0
        """
        if X.shape[0] < self.window:
            return -1
        try:
            X_slice = X[-1-self.window+1 : ]
            no_of_faulty_params  = 0
            no_of_healthy_params = 0

            for feature in self.profile_cutoffs.keys():
                matches = stumpy.match(Q=X_slice[feature], T=self.faulty[feature])
                score = matches[0][0]

                if score < self.profile_cutoffs[feature]:
                    no_of_faulty_params += 1
                else:
                    no_of_healthy_params += 1

            if no_of_faulty_params > 3:
                return 1 # Faulty
            else:
                return 0 # Healthy
            
        except Exception as e:
            return -1


    def calculate_f1(self, data_slice):
        
        true = []
        predictions = []

        for i in range(data_slice.shape[0]):
            try:
                status = data_slice.iloc[i]['status']
                prediction = self.predict(data_slice.iloc[:i])

                if prediction != -1:
                    predictions.append(prediction)
                    true.append(status)
                    
            except Exception as e:
                pass
        return (predictions, true)
    
    def callback_update_f1(self, future):
        try:
            predictions, true = future.result()

            for i in range(len(predictions)):
                if predictions[i] == true[i]:
                    if true[i] == 1:
                        self.TP += 1
                    else:
                        self.TN += 1
                else:
                    if true[i] == 1:
                        self.FN += 1
                    else:
                        self.FP += 1 
        except Exception as e:
            pass




    def check_performance(self, data):
        """
        This method calculates the TP, TN, FP, FN scores on the parameter - data
        1. Divides the data in chunks
        2. Start a new process with each chunk for calculation
        """
        with ProcessPoolExecutor() as executor:
            query_N = data.shape[0]
            chunk = 500
            start_idx = 0
            end_idx = chunk

            while True:
                if end_idx > query_N:
                    end_idx = query_N
                if start_idx == end_idx:
                    break
                
                job = executor.submit(self.calculate_f1, data.iloc[start_idx : end_idx])
                job.add_done_callback(self.callback_update_f1)

                start_idx, end_idx = end_idx, end_idx + chunk
            
            executor.shutdown()

        

