from training.clustering import Clustering
from training.isolation_forest import Isolation
from training.matrix_profile import MatrixProfiling
import pandas as pd
from path.path import PROCESSED_DATA, CLUSTERING_MODEL, PROFILE_MODEL, ISOLATION_MODEL
import pickle
from log.logging import LOGGER
import logging
from datetime import datetime


def train_clustering():
    """
    1. First try clustering model with "ewm" and "sma" and different values of rolling_window
    """

    try:
        preprocessed =  pd.read_csv(PROCESSED_DATA, index_col='timestamp')
        preprocessed.index = pd.to_datetime(preprocessed.index) # Convert index to datetime

        best_recall_score = 0
        best_model = None

        LOGGER.log_clustering(message="Start of training of clustering", level=logging.INFO)

        for rolling_window in [50,100,150,200,250]:
            for method in ["ewm", "sma"]:
                model = Clustering(rolling_window, method, 0.5)
                model.fit(preprocessed)

                LOGGER.log_clustering(f"Rolling window - {rolling_window}, method - {method}, test recall - {model.test_recall : 0.3f}, test precision - {model.test_precision : 0.3f}, test f1 - {model.test_f1 : 0.3f} ", logging.INFO)

                if model.test_recall > best_recall_score:
                    best_model = model
                    best_recall_score = model.test_recall
        
        if best_model is not None:
            # Save the model
            with open(CLUSTERING_MODEL, 'wb') as f:
                pickle.dump(best_model, f)
                f.close()
            LOGGER.log_clustering(f"Best model rolling window - {best_model.rolling_window}, method - {best_model.method}, test recall - {best_model.test_recall : 0.3f}, test precision  - {best_model.test_precision:0.3f}, test f1 - {best_model.test_f1 : 0.3f}", logging.INFO)
            LOGGER.log_clustering(message="Successful end of clustering\n\n", level=logging.INFO)
    
    except Exception as e:
        LOGGER.log_clustering(message=f"Error in training clustering model {e}\n\n", level=logging.ERROR)


def train_isolation():
    """
    1. Try Isolation forest with different values of rolling window, contamination and max_features
    """
    try:
        preprocessed =  pd.read_csv(PROCESSED_DATA, index_col='timestamp')
        preprocessed.index = pd.to_datetime(preprocessed.index) # Convert index to datetime

        best_f1_score = 0
        best_model = None

        LOGGER.log_isolation(message="Start of training of isolation forest", level=logging.INFO)

        for rolling_window in [100,150,200]:
            for method in ["ewm", "sma"]:
                for contamination in [0.05, 0.07, 0.09, 0.1]:
                    for max_feature in [0.5,0.7,0.9]:
                        model = Isolation(rolling_window, method, 0.5, contamination, max_feature)
                        model.fit(preprocessed)
                        LOGGER.log_isolation(f"Rolling window - {rolling_window}, method - {method}, contamination - {contamination}, max_feature - {max_feature}, test recall - {model.test_recall : 0.3f}, test precision {model.test_precision : 0.3f}, test f1 {model.test_f1 : 0.3f}", logging.INFO)

                        if model.test_f1 > best_f1_score:
                            best_model = model
                            best_f1_score = model.test_f1
        
        if best_model is not None:
            # Save the model
            with open(ISOLATION_MODEL, 'wb') as f:
                pickle.dump(best_model, f)
                f.close()
            LOGGER.log_isolation(f"""Best model rolling window - {best_model.rolling_window}, method - {best_model.method}, contamination - {best_model.contamination}, max_features - {best_model.max_features}, test recall - {best_model.test_recall : 0.3f}, test precision- {best_model.test_precision : 0.3f}, test f1 - {best_model.test_f1 : 0.3f}""", logging.INFO)
            LOGGER.log_isolation(message="Successful end of isolation forest\n\n", level=logging.INFO)
    
    except Exception as e:
        LOGGER.log_isolation(f"Error in training isolation forest model {e}\n\n", logging.ERROR)





def do_matrix_profiling():
    try:
        data =  pd.read_csv(PROCESSED_DATA, index_col='timestamp')
        data.index = pd.to_datetime(data.index) # Convert index to datetime

        mat_profile = MatrixProfiling()
        mat_profile.find_cutoffs(data)

        with open(PROFILE_MODEL, 'wb') as f:
            pickle.dump(mat_profile, f)
            f.close()

        LOGGER.log_profiling(f"Profiling done successfully and model saved", logging.INFO)
        LOGGER.log_profiling(mat_profile.profile_cutoffs, logging.INFO)

    except Exception as e:
        LOGGER.log_profiling(f"Error in finding matrix profiling - {e}", logging.ERROR)

    
def do_sample_prediction_profiling(start_date, end_date):
    try:
        data =  pd.read_csv(PROCESSED_DATA, index_col='timestamp')
        data.index = pd.to_datetime(data.index) # Convert index to datetime

        LOGGER.log_profiling(f"Doing prediction on data from  {start_date} to {end_date}", logging.INFO)
        test_data = data.loc[start_date : end_date]
        

        with open(PROFILE_MODEL, 'rb') as f:
            profile_model = pickle.load(f)
        
        profile_model.check_performance(test_data)

        LOGGER.log_profiling(f"True Positives - {profile_model.TP}, True Negative - {profile_model.TN}, False Positives - {profile_model.FP}, False Negatives - {profile_model.FN}", logging.INFO)


    except Exception as e:
        LOGGER.log_profiling(f"Error in sample predictions - {e}", logging.ERROR)