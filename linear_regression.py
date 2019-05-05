import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return b_0, b_1


def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m",
                marker="o", s=30)

    # predicted response vector
    y_pred = b[0] + b[1] * x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()


# load data
df = pd.read_csv("przetworzone_dane2.csv",  delimiter=";")

df_number_col = np.array(list(map(int, np.array(df.iloc[:, 1:2]))))

x = np.array(list(range(1, len(df_number_col) + 1)))
y = df_number_col

# estimating coefficients
b = estimate_coef(x, y)
print("Estimated coefficients:\nb_0 = {}  \
\nb_1 = {}".format(b[0], b[1]))

# plotting regression line
plot_regression_line(x, y, b)