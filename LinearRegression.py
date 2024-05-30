import csv
import numpy as np

class LinearRegression:
    def __init__(self, file_path, shape, lr=0.1, normalization_type='MIN_MAX'):
        self.x  = None
        self.n_x  = None
        self.w = np.zeros(shape)
        self.b = 0
        self.y = None
        self.file = file_path
        self.x_max = None
        self.x_min = None
        self.lr = lr
        print(normalization_type)
        self.normalization_type = normalization_type

    def reading_data(self):
        with open(self.file, 'r') as csvfile:
            csvReader = csv.DictReader(csvfile)
            for row in csvReader:
                x = []
                for element in row:
                        x.append(float(row[element]))

                y = x[-1]
                x.pop()
                self.x =  np.array([x]) if self.x is None else np.concatenate((self.x, np.array(x).reshape(1, len(x))), axis=0)
                self.y =  np.array([y]) if self.y is None else np.concatenate((self.y, np.array([y])), axis=0)


                if self.normalization_type == 'MIN_MAX':
                    self.x_max = np.array(x) if self.x_max is None else np.array([max(a,b) for a,b in zip(self.x_max, np.array(x))])
                    self.x_min = np.array(x) if self.x_min is None else np.array([min(a,b) for a,b in zip(self.x_min, np.array(x))])


    # here we are using Liner normalization or "Max-Min"
    def normalization(self):
        
        if self.normalization_type == 'None':
            self.n_x = x
        if self.normalization_type == 'MIN_MAX':
            i = 0
            for x in self.x.T:
                d = (self.x_max[i] - self.x_min[i])
                x_min = self.x_min[i]
                if self.n_x is None:
                    self.n_x = ((x - x_min) / d).reshape(-1, 1)
                else:
                    self.n_x = np.concatenate((self.n_x, (x - x_min ) / d)).reshape(-1, i + 1, order='F')
                i += 1
        if self.normalization_type == 'Z_SCORE':
            i = 0
            for x in self.x.T:
                std = x.std()
                mean = x.mean()
                if self.n_x is None:
                    self.n_x = ((x - mean) / std)
                else:
                    self.n_x = np.concatenate((self.n_x, (x - mean) / std)).reshape(-1, i + 1, order='F')
                i += 1



    def f(self, x):
        return (x.dot(self.w) + self.b)

    #the learning phase
    def learning(self):
        y = self.y
        b = 0.01
        lr = self.lr
        n = len(self.x)
        self.w = self.w.reshape(-1, 1)
        self.y = self.y.reshape(-1, 1)

        # print(self.n_x.T.shape)
        # for x in self.n_x.T:
        #     print(x.reshape(-1, 1).shape)
 
        # print(sum(((self.n_x.dot(self.w) + self.b - self.y) * self.n_x.T[0].reshape(-1, 1))) /n)

        for _ in range(10000):
            w = np.copy(self.w)
            for i in range(len(w)):
                w[i] -= lr * sum(((self.n_x.dot(self.w) + self.b - self.y) * self.n_x.T[i].reshape(-1, 1))) /n
            self.b -= lr * sum((self.n_x.dot(self.w) + self.b) - self.y) / n
            self.w = w
        w = np.copy(self.w)
        print(w)
        if self.normalization_type == 'MIN_MAX':
            for i in range(len(w)):
                self.w[i] = (self.w[i] / (self.x_max[i] - self.x_min[i]))
            self.b = self.b  - (w[0] * self.x_min[0] / (self.x_max[0] - self.x_min[0]))
    
        print((self.w.dot(240000) + self.b))
        print(self.x.shape, self.w.shape)
        
        print(sum(((self.x.dot(self.w) + self.b - self.y) ** 2 )) / n)
        # # print(w24000)
        
