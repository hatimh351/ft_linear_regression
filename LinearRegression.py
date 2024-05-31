import csv
import numpy as np

class LinearRegression:
    def __init__(self, file_path, lr=0.1, normalization_type='None', iteration=1000):
        self.x  = None
        self.n_x  = None
        self.w = None
        self.b = 0
        self.y = None
        self.file = file_path
        self.x_max = None
        self.x_min = None
        self.x_std = None
        self.x_mean = None
        self.lr = lr
        self.normalization_type = normalization_type
        self.done = False
        self.iteration = iteration

    def reading_data(self):
        with open(self.file, 'r') as csvfile:
            csvReader = csv.DictReader(csvfile)
            for row in csvReader:
                x = []

                for element in row:
                        try:
                            x.append(float(row[element]))
                        except:
                            pass
                y = x[-1]
                x.pop()
                self.x =  np.array([x]) if self.x is None else np.concatenate((self.x, np.array(x).reshape(1, len(x))), axis=0)
                self.y =  np.array([y]) if self.y is None else np.concatenate((self.y, np.array([y])), axis=0)


                if self.normalization_type == 'MIN_MAX':
                    self.x_max = np.array(x) if self.x_max is None else np.array([max(a,b) for a,b in zip(self.x_max, np.array(x))])
                    self.x_min = np.array(x) if self.x_min is None else np.array([min(a,b) for a,b in zip(self.x_min, np.array(x))])

    
    def normalization(self):
        if self.x is None or self.y is None:
            print('we didn\'t  read any data')
            exit(0)
        if self.normalization_type == 'None':
            self.n_x = x
        if self.normalization_type == 'MIN_MAX':
            i = 0
            for x in self.x.T:
                d = (self.x_max[i] - self.x_min[i])
                x = x.reshape(-1, 1)
                x_min = self.x_min[i]
                if self.n_x is None:
                    self.n_x = ((x - x_min) / d).reshape(-1, 1)
                else:
                    self.n_x = np.concatenate((self.n_x, (x - x_min ) / d), axis=1).reshape(-1, i + 1, order='F')
                i += 1
        if self.normalization_type == 'Z_SCORE':
            i = 0
            for x in self.x.T:
                std = x.std()
                mean = x.mean()
                x = x.reshape(-1, 1)

                self.x_std = np.array([std]) if self.x_std  is None else np.append(self.x_std, std)
                self.x_mean = np.array([mean]) if self.x_mean is None else np.append(self.x_mean, mean)
                if self.n_x is None:
                    self.n_x = ((x - mean) / std).reshape(-1, 1)
                else:
                    self.n_x = np.concatenate((self.n_x, (x - mean) / std), axis=1).reshape(-1, i + 1, order='F')
                i += 1



    def f(self, x):
        return (x.dot(self.w) + self.b)

    #the learning phase
    def learning(self):
        y = self.y
        lr = self.lr
        n = len(self.x)
        self.y = self.y.reshape(-1, 1)
        self.w = np.zeros((self.n_x.shape[1], 1))
        

        for _ in range(self.iteration):
            w = np.copy(self.w)
            for i in range(len(w)):
                w[i] -= lr * sum(((self.n_x.dot(self.w) + self.b - self.y) * self.n_x.T[i].reshape(-1, 1))) / n
            self.b -= lr * sum((self.n_x.dot(self.w) + self.b) - self.y) / n
            self.w = w
        w = np.copy(self.w)

        if self.normalization_type == 'MIN_MAX':
            for i in range(len(w)):
                self.w[i] = (self.w[i] / (self.x_max[i] - self.x_min[i]))
            self.b = self.b  - sum(w.reshape(-1,) * self.x_min / (self.x_max - self.x_min))
    

        if self.normalization_type == 'Z_SCORE':
            for i in range(len(w)):
                self.w[i] = (w[i] / (self.x_std[i]))
            self.b = self.b  - sum((w.reshape(-1,) * self.x_mean / self.x_std))


        self.done = True


    def info(self):
        infos = {}
        infos['learning rate'] = self.lr
        infos['Normalization used'] = self.normalization_type
        infos['Iterations for learning'] = self.iteration

        for _info in infos:
            print(f"{_info} : {infos[_info]}")





    def __del__(self):
        if not self.done:
            return None
        with open('prediction_coefficient', 'w') as file:
            result = []
            for w in self.w:
                result.append(str(w[0]))
            result.append(str(self.b[0]))
            file.write(','.join(result))
        print("This project is Done by Miloki !")