import csv
import numpy as np
from matplotlib import pyplot as plt

class LinearRegression:
    def __init__(self, file_path, shape, lr=0.1):
        self.x  = np.empty([0, shape])
        self.n_x  = np.empty([0, shape])
        self.n_y  = np.empty([1, 0])
        self.w = np.zeros(shape)
        self.b = np.zeros(1)
        self.y = np.empty([0,1])
        self.file = file_path
        self.x_max = None
        self.x_min = None
        self.lr = lr

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

                self.x_max = np.array(x) if self.x_max == None else np.array([max(a,b) for a,b in zip(self.x_max, np.array(x))])
                self.x_min = np.array(x) if self.x_min == None else np.array([min(a,b) for a,b in zip(self.x_min, np.array(x))])



    # here we are using Liner normalization or "Max-Min"
    def normalization(self):
        d = 1 / (self.x_max - self.x_min)
        
        for x in self.x:
            self.n_x = np.append(self.n_x, [(x - self.x_min)]  * d, axis=0)


    def f(self, x):
        return (x.dot(self.w) + self.b)

    #the learning phase
    def learning(self):
        x = self.n_x
        y = self.y
        b = self.b
        lr = self.lr
        n = len(self.x)

        n_w = len(self.w)
        for _ in range(10000):
            w = np.copy(self.w)
            for i in range(n_w):
                # print(x[i])
                w[i] -= lr * sum((self.f(x) - y) * x.reshape((24,))) * 1/n
            
            b -= lr * sum(self.f(x) - y) * 1/n
            self.w  = w


        w = np.copy(self.w)
        self.w = (self.w / (self.x_max - self.x_min))
        self.b = self.b  - (w * self.x_min / (self.x_max - self.x_min))

        print(self.b +  self.w * 240000)

    # the result 
    def linearRegression():
        pass


