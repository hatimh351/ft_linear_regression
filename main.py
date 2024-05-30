import sys
from  LinearRegression import LinearRegression

def main():
    if len(sys.argv) != 2:
        return
    Lr = LinearRegression(sys.argv[1], 1, normalization_type='MIN_MAX')
    Lr.reading_data()
    Lr.normalization()
    Lr.learning()


if '__main__' == __name__:
    main()