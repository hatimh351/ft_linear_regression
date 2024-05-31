import numpy as np
import sys

def predection(w, b, x):
    try:
        result = w.dot(x) + b
        print(result)
    except:
        print("Can't predict using the informations provided ")
        exit(1)



if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print("provide a set of x delimited by space in one argv" )
        exit(0)
    try :
        sys.argv[1] = sys.argv[1].strip()
        X = sys.argv[1].split(' ')
        X = np.array([np.float128(x) for x in X])
    except:
        print("provide a set of x delimited by space in one argv" )
        exit(0)

    try:    
        with open('prediction_coefficient', 'r') as file:
            coefficients = file.read()
            w = coefficients.split(',')
            w = list(map(lambda x: float(x), w))
            b =  np.float128(w.pop())
            w = np.array([np.float128(o) for o in w])
    except:
        print("Problem while reading the file")
        exit(1)

    predection(w,b, X)
