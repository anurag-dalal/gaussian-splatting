import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import time

def distance(end):
    return (end[0]*end[0] + end[1]*end[1])**0.5


ns = [i*100 for i in range(1,100)]
l = 10
prev = [0, 0]
ds = []
for n in ns:
    d = []
    for i in range(50):
        for i in range(n):
            end = [prev[0], prev[1]]
            direction = random.randint(0,3)
            if direction==0:
                end = [prev[0], prev[1] + l]
            elif direction==1:
                end = [prev[0], prev[1] - l]
            elif direction==2:
                end = [prev[0] + l, prev[1]]
            else:
                end = [prev[0] - l, prev[1]]
            ypoints = np.array([prev[0], end[0], prev[1], end[1]])
            prev = [end[0], end[1]]
        d.append(distance(end))
    ds.append(sum(d)/len(d))
plt.plot(np.array(ns), np.array(ds))
plt.show()
plt.savefig("output.jpg")