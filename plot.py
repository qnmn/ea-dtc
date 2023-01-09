#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

wine = np.genfromtxt('wine_history.csv', delimiter=',')
glass = np.genfromtxt('glass_history.csv', delimiter=',')

plt.figure(figsize=(3,3))
plt.plot(wine[:,0], wine[:,1])
plt.xlabel('Dataset size')
plt.ylabel('Fitness reached')
plt.ylim(bottom=0.9, top=1.0)
plt.savefig('wine_history.png', bbox_inches='tight')


plt.figure(figsize=(3,3))
plt.plot(glass[:,0], glass[:,1])
plt.xlabel('Dataset size')
plt.ylabel('Fitness reached')
plt.ylim(bottom=0.7, top=1.0)
plt.savefig('glass_history.png', bbox_inches='tight')
