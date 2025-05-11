#logger python documentation : https://docs.python.org/3/library/logging.html
#logger is the purpose of that any execution that probably happens that should be able to log all those information and exection in some files that will be able to track some errors even the some custom exception errors we will try to log that text file. so that we need to implement logger.
import logging
import os
from datetime import datetime

#create a log file
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE) #we need to give a path for the log file also
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s]%(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    

)

'''if __name__=="__main__":
    logging.info("Logging has started")'''