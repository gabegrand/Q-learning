import model
import random

# Number of learning trials
TRIALS = 1000

# Positive probabilities for stimuli pairs
ProbA = 0.8
ProbB = 0.2

ProbC = 0.7
ProbD = 0.3

ProbE = 0.6
ProbF = 0.4

# Weighted probabilistic pos/neg choice
def flip(p):
    return int(random.random() < p)

# TESTING
learner = model.QLearner()

# Learning sequence
for i in xrange(TRIALS):
    learner.learn("A", flip(ProbA))

for i in xrange(TRIALS):
    learner.learn("B", flip(ProbB))

for i in xrange(TRIALS):
    learner.learn("C", flip(ProbC))

for i in xrange(TRIALS):
    learner.learn("D", flip(ProbD))

for i in xrange(TRIALS):
    learner.learn("E", flip(ProbE))

for i in xrange(TRIALS):
    learner.learn("F", flip(ProbF))

print "Learned A:", learner.getQ("A")
print "Learned B:", learner.getQ("B")
print "Learned C:", learner.getQ("C")
print "Learned D:", learner.getQ("D")
print "Learned E:", learner.getQ("E")
print "Learned F:", learner.getQ("F")