from lib.OVO import *


if __name__ == "__main__":
    my_OVO = OVO(3)
    my_OVO.train("../data/Iris/iris.csv")
    my_OVO.test("../data/Iris/iris_test.csv")