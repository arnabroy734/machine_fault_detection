from preprocessing.preprocessor import Preprocessor
from training.training import train_clustering, do_matrix_profiling, do_sample_prediction_profiling, train_isolation
from datetime import datetime

if __name__=="__main__":
     Preprocessor().preprocess()
     train_clustering()
     train_isolation()
     # do_matrix_profiling()
     # do_sample_prediction_profiling(start_date=datetime(2020, 4, 17, 20, 0), end_date=datetime(2020, 4, 19, 5, 0))


    