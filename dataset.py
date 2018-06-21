"""Implements a dataset class for handling image data"""

DATA_PATH_BASE = ''

class Dataset(object):
  def __init__(self, data_list_file=None, process_func=None):
    """
      Args:
    """
    self.data_list_file = data_list_file
 
  def read_data_list_file(self):
    """Reads the data list_file into python list
    """
    f = open(self.data_list_file)
    data_list =  [DATA_PATH_BASE+line.rstrip() for line in f]
    self.data_list = data_list
    
    return data_list
