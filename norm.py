import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import norm

from matplotlib.font_manager import FontProperties
fonts = FontProperties(fname = "/Library/Fonts/Kaiti.ttc", size=14)

import warnings
warnings.filterwarnings("ignore")

# 计算数据
pp = np.linspace(0, 2, 20)
plt.figure(figsize=(12, 15))
for ii in np.arange(len(pp)):
    p = pp[ii]
    x1f = np.linspace(-1, 0, 50)
    x1z = np.linspace(0, 1, 50)
    x1 = np.concatenate((x1f, x1z))
    x2 = (1 - abs(x1)**p)**(1/p)
    x2f = -(1 - abs(x1)**p)**(1/p)

    plt.subplot(5, 4, ii + 1)
    plt.plot(x1, x2, "r", x1, x2f, "r")
    plt.arrow(-2, 0, 4, 0)
    plt.arrow(0, -2, 0, 4)
    plt.axis("equal")
    plt.xlabel("")
    plt.ylabel("")
    plt.title("$L_p$ norm " + "p=%.2f" % p)
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))

# 查看图像
plt.subplots_adjust(bottom = 0.05)
plt.suptitle("$L_p$ norm p = [0 ~ 2]", x= 0.5, y = 0.92, size = 16)
plt.show()

# 计算数据
pp = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 100, np.inf])
plt.figure(figsize=(10, 9))
for ii in np.arange(len(pp)):
    p = pp[ii]
    x1f = np.linspace(-1, 0, 50)
    x1z = np.linspace(0, 1, 50)
    x1 = np.concatenate((x1f, x1z))
    x2 = (1 - abs(x1)**p)**(1/p)
    x2f = -(1 - abs(x1)**p)**(1/p)

    plt.subplot(3, 3, ii + 1)
    if (p != 0) & (p != np.inf):
        plt.plot(x1, x2, "r", x1, x2f, "r")
    else:
        plt.plot(x1, x2, "r", x1, x2f, "r")
        plt.plot(x2, x1, "r", x2f, x1, "r")
    plt.arrow(-2, 0, 4, 0)
    plt.arrow(0, -2, 0, 4)
    plt.axis("equal")
    plt.xlabel("")
    plt.ylabel("")
    plt.title("$L_p$ norm " + "p=%.2f" % p)
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))

# 查看图像
plt.subplots_adjust(bottom = 0.05)
plt.suptitle("$L_p$ norm p = [0 ~ inf]", x= 0.5, y = 0.93, size = 16)
plt.show()

# 计算数据
pp = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 1000, np.inf])
plt.figure(figsize=(5, 5))
plt.arrow(-2, 0, 4, 0)
plt.arrow(0, -2, 0, 4)
plt.axis("equal")
plt.xlabel("x1")
plt.ylabel("x2")
plt.xlim((-2, 2))
plt.ylim((-2, 2))
for ii in np.arange(len(pp)):
    p = pp[ii]
    x1f = np.linspace(-1, 0, 50)
    x1z = np.linspace(0, 1, 50)
    x1 = np.concatenate((x1f, x1z))
    x2 = (1 - abs(x1)**p)**(1/p)
    x2f = -(1 - abs(x1)**p)**(1/p)
    plt.plot(x1, x2, "r", x1, x2f, "r")

plt.title("$L_p$ norm p = [0 ~ inf]", size=16)
plt.show()

# 计算数据
pp = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 1000, np.inf])
for ii in np.arange(len(pp)):
    p = pp[ii]
    x1f = np.linspace(-1, 0, 50)
    x1z = np.linspace(0, 1, 50)
    x1 = np.concatenate((x1f, x1z))
    x2 = (1 - abs(x1)**p)**(1/p)
    x2f = -(1 - abs(x1)**p)**(1/p)
    plt.plot(x1, x2, "r", x1, x2f, "r")

plt.xlim((-2, 2))
plt.ylim((-2, 2))

plt.title("$L_p$ norm p = [0 ~ inf]", size=16)
plt.show()