# TESTING SUITE

import train as tn
import random
import numpy as np
import matplotlib.pyplot as plt

# One-off A vs. B test
def simple_test(learner):
    print "CHOICE (A vs. B):", learner.choose("A","B")
    print "CHOICE (C vs. D):", learner.choose("C","D")
    print "CHOICE (E vs. F):", learner.choose("E","F")

# Test accuracy over a number of trials
def test(learner, trials):
    chosen_A = 0
    for i in xrange(trials):
        choice = learner.choose("A","B")
        if choice == "A": chosen_A += 1
    return float(chosen_A)/trials

# Varying number of training trials
stupid = tn.train(0.3, 0.3, 0.2, 4)
medium = tn.train(0.3, 0.3, 0.2, 40)
smart = tn.train(0.3, 0.3, 0.2, 400)

'''
# Visualizing effect of training on performance
N = 3
performance = (test(stupid, 100),
               test(medium, 100),
               test(smart, 100))

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, performance, width, color='g')

# add some text for labels, title and axes ticks
ax.set_ylabel('Accuracy')
ax.set_title('Effect of training on performance')
ax.set_xticks(ind + width)
ax.set_xticklabels(('Stupid', 'Medium', 'Smart'))

ax.legend((rects1[0], ('Choose A Task'))
plt.show()
'''

performance = (test(stupid, 100),
               test(medium, 100),
               test(smart, 100))
print performance


'''
# Varying BETA:
print "\nLOW BETA"
low_beta = train(0.3, 0.3, 0.1, 1000)
test(low_beta, 100)

print "\nMED BETA"
med_beta = train(0.3, 0.3, 0.5, 1000)
test(med_beta, 100)

print "\nHIGH BETA"
high_beta = train(0.3, 0.3, 0.8, 1000)
test(high_beta, 100)
'''
