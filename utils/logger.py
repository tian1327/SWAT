import logging
import os
from datetime import date

def get_logger(dir_path, file_name, log_mode='file'):

    logger = logging.getLogger('')
    
    # Set the logging level
    logger.setLevel(logging.INFO)
    
    # Create a file handler and set its level
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f'Created directory: {dir_path}')

    # Create a log message formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if log_mode in ['file', 'both']:
        file_handler = logging.FileHandler(f'{dir_path}/{file_name}.log', 
                                            # mode='a',
                                            mode='w',
                                            )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if log_mode in ['console', 'both']:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    return logger
