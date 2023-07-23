import logging


filenames = dict(
    CLUSTERING_LOGS = "./log/clustering_log.txt",    
    PROFILING_LOGS = "./log/profiling_log.txt",
    PREDICTION_LOGS = "./log/prediction_log.txt",
    PREPROCESSING_LOGS = "./log/preprocessing_logs.txt",
    ISOLATION_FOREST_LOGS = "./log/isolation_forest_logs.txt",
    STARTUP_LOGS = "./log/startup_logs.txt"
    
)



class AppLogger:
    """
    Description:
    Simple class to save logs into multiple files
    """
    def __init__(self):
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s]')
        self.handlers = {}

        for filename in filenames.keys():
            self.handlers[filename] = logging.FileHandler(filenames[filename])
            self.handlers[filename].setFormatter(formatter)
        


    def custom_log(self, filename, level=logging.INFO):
        custom_logger = logging.getLogger(filenames[filename])
        custom_logger.setLevel(level)
        custom_logger.addHandler(self.handlers[filename])
        return custom_logger

    def write_log(self, logger, message, level):
        if level == logging.INFO:
            logger.info(message)
        elif level == logging.ERROR:
            logger.error(message)
    
    def log_clustering(self, message, level):
        clustering_logger = self.custom_log("CLUSTERING_LOGS")
        self.write_log(clustering_logger, message, level)

    def log_profiling(self, message, level):
        profiling_logger = self.custom_log("PROFILING_LOGS")
        self.write_log(profiling_logger, message, level)


    def log_prediction(self, message, level):
        prediction_loggger = self.custom_log("PREDICTION_LOGS")
        self.write_log(prediction_loggger, message, level)
    
    def log_preprocessing(self, message, level):
        preprocessing_logger = self.custom_log("PREPROCESSING_LOGS")
        self.write_log(preprocessing_logger, message, level)

    def log_isolation(self, message, level):
        isolation_logger = self.custom_log("ISOLATION_FOREST_LOGS")
        self.write_log(isolation_logger, message, level)
    
    def log_startup(self, message, level):
        startup_log = self.custom_log('STARTUP_LOGS')
        self.write_log(startup_log, message, level)


LOGGER = AppLogger()
