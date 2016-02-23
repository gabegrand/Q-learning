# Choose A vs. Avoid B Test

import train as tn
import random
import numpy as np
import matplotlib.pyplot as plt

# Accuracy in choosing positive stimulus
def chooseA_test(learner, trials):
    # Set of more neutral stimuli
    others = ["C", "D", "E", "F"]
    chosen_A = 0
    for i in xrange(trials):
        choice = learner.choose("A",random.choice(others))
        if choice == "A": chosen_A += 1
    return float(chosen_A)/trials

# Accuracy in avoiding negative stimulus
def avoidB_test(learner, trials):
    # Set of more neutral stimuli
    others = ["C", "D", "E", "F"]
    avoidedB = 0
    for i in xrange(trials):
        choice = learner.choose("B",random.choice(others))
        if choice != "B": avoidedB += 1
    return float(avoidedB)/trials

# Varying ALPHA_POS and ALPHA_NEG:
optimist = tn.train(0.4, 0.15, 0.2, 1000)
pessimist = tn.train(0.15, 0.4, 0.2, 1000)
neutral = tn.train(0.3, 0.3, 0.2, 1000)

# Visualizing the Choose A / Avoid B test for different alphas
N = 3
chooseA = (chooseA_test(optimist, 100),
           chooseA_test(pessimist, 100),
           chooseA_test(neutral, 100))

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, chooseA, width, color='g')

chooseB = (avoidB_test(optimist, 100),
           avoidB_test(pessimist, 100),
           avoidB_test(neutral, 100))

rects2 = ax.bar(ind + width, chooseB, width, color='r')

# add some text for labels, title and axes ticks
ax.set_ylabel('Accuracy')
ax.set_title('Comparative accuracy on choice/avoidance tasks')
ax.set_xticks(ind + width)
ax.set_xticklabels(('Optimist', 'Pessimist', 'Neutral'))

ax.legend((rects1[0], rects2[0]), ('Choose A Task', 'Avoid B Task'))
plt.show()
