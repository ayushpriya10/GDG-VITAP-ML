import numpy as np
import matplotlib.pyplot as plt

def estimateCoeffs(x, y):
    n = np.size(x)

    mx = np.mean(x)
    my = np.mean(y)

    covarianceXY = np.sum(y*x - n*my*mx)
    varianceX = np.sum(x*x - n*mx*mx)

    w = covarianceXY / varianceX
    b = my - w*mx

    return (b, w)

def plotRegressionLine(x, y, b, w):
    plt.scatter(x, y, color = "m", marker = "o", s = 30)

    y_pred = w*x + b

    plt.plot(x, y_pred, color = "g")

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()

def main():
    xData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    yData = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    #xData = [2, 4, 5, 6.5, 9, 12, 13, 13.5, 15, 17]
    #yData = [3, 6, 7.5, 9.75, 13.5, 18, 19.5, 20.25, 22.5, 25.5]

    x = np.array(xData)
    y = np.array(yData)

    b, w = estimateCoeffs(x, y)
    print("Estimated coefficients:\nW = {}\nB = {}".format(w, b))

    plotRegressionLine(x, y, b, w)

    #newX = int(input("Enter Value to predict new Y: "))
    #print("Given X: {}\nPredicted Y: {}".format(newX, (newX*w + b)))

if __name__ == "__main__":
    main()
