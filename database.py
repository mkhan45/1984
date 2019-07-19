from collections import defaultdict
import pickle
from pathlib import Path
def create_or_reset(databasepath):
    '''
    Resets or creates given descriptor database
    Parameters:
    ---------------
    databasepath: Path object
    
    Returns:
    ---------------
    None
    '''
    
    
    dict1 = dict()
    with open(databasepath, 'wb') as handle:
        pickle.dump(dict1, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def check_contents(databasepath):
    '''
    Returns list of names already in the database
    Parameters:
    ---------------
    databasepath: Path object
    
    Returns:
    ---------------
    namelist: list object
    '''
    with open(databasepath, mode = 'rb') as handle:
        database = pickle.load(handle)
    print(database.keys())
