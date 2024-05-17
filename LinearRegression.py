import csv
import numpy as np

class LinearRegression:
    def __init__(self, file_path):
        self.x  = np.arry([])
        self.b = np.arry([])
        self.y = None
        self.file = file_path

    #Letting the data be normalized to avoid overfitting
    def normalization(self):
        # reading the file
        with open(self.file, 'r') as csvfile:
            csvReader = csv.DictReader(csvfile)
            for row in csvReader:
                pass
    #the learning phase
    def learning():
        pass

    # the result 
    def linearRegression():
        pass


