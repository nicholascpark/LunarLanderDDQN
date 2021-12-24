import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def geometric(rate):

    geo = []
    n = 0
    x = 1
    while n < 3000:
        geo.append(x)
        x *= rate
        n+=1
    return geo

plt.figure(0)
plt.title("Geometric Decay Rate Visualized")
colorindex = 0
geometric_decay = [0.9999, 0.9995, 0.999, 0.995, 0.99, 0.95, 0.9]
for i in geometric_decay:
    geo = geometric(i)
    plt.plot(geo, color = cm.jet(colorindex), linewidth = 1.5, label = r"$rate = $" + str(i))
    colorindex += 1/len(geometric_decay)
plt.xlim(-5, 500)
plt.grid(linewidth=0.2)
plt.xlabel("Episode")
plt.ylabel("Epsilon")
# plt.xscale("log")
plt.legend(fontsize="small")
plt.show()
plt.savefig("geometric_decay_rate")

