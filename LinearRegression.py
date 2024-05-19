import csv
import numpy as np

class LinearRegression:
    def __init__(self, file_path, shape):
        self.x  = np.empty([0, shape])
        self.n_x  = np.empty([0, shape])
        self.b = np.array([])
        self.y = np.empty([0,1])
        self.file = file_path
        self.max = None
        self.min = None



    def reading_data(self):
        with open(self.file, 'r') as csvfile:
            csvReader = csv.DictReader(csvfile)
            for row in csvReader:
                x = []
                y = None
                for element in row:
                    if element.startswith('x'):
                        x.append(int(row[element]))
                    else:
                        y = int(row[element])
                        
                self.x = np.append(self.x, [x], axis=0)
                self.y = np.append(self.y, y)
                self.max = np.array(x) if self.max == None else np.array([max(a,b) for a,b in zip(self.max, np.array(x))])
                self.min = np.array(x) if self.min == None else np.array([min(a,b) for a,b in zip(self.min, np.array(x))])



    # here we are using Liner normalization or "Max-Min"
    def normalization(self):
        d = 1 / (self.max - self.min)
        for x in self.x:
            self.n_x = np.append(self.n_x, [(x - self.min)]  * d, axis=0)


    #the learning phase
    def learning():
        pass

    # the result 
    def linearRegression():
        pass


