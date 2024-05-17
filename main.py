import sys
from  LinearRegression import LinearRegression

def main():
    if len(sys.argv) != 2:
        return
    Lr = LinearRegression(sys.argv[1])
    Lr.normalization()


if '__main__' == __name__:
    main()