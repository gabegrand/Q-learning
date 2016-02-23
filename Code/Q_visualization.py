# Visualizing the Q Learning process

import train as tn
import random
import numpy as np
import matplotlib.pyplot as plt

N = 6
learner = tn.train(0.001, 0.001, 0.2, 10000)
QValues = (learner.getQ("A"),
           learner.getQ("B"),
           learner.getQ("C"),
           learner.getQ("D"),
           learner.getQ("E"),
           learner.getQ("F"))

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, QValues, width, color='b')

groundTruth = (tn.ProbA, tn.ProbB, tn.ProbC, tn.ProbD, tn.ProbE, tn.ProbF)

rects2 = ax.bar(ind + width, groundTruth, width, color='g')

# add some text for labels, title and axes ticks
ax.set_ylabel('Accuracy')
ax.set_title('Learned Q Values vs. Ground Truth')
ax.set_xticks(ind + width)
ax.set_xticklabels(('A', 'B', 'C', 'D', 'E', 'F'))

ax.legend((rects1[0], rects2[0]), ('Learned Q Value', 'Ground Truth Probability'))
plt.show()
