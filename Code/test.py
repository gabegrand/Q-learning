import model
import random

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
def test(pos, neg, beta, trials):
    learner = model.QLearner(pos, neg, beta)

    # Learning sequence
    for i in xrange(trials):
        learner.learn("A", flip(ProbA))

    for i in xrange(trials):
        learner.learn("B", flip(ProbB))

    for i in xrange(trials):
        learner.learn("C", flip(ProbC))

    for i in xrange(trials):
        learner.learn("D", flip(ProbD))

    for i in xrange(trials):
        learner.learn("E", flip(ProbE))

    for i in xrange(trials):
        learner.learn("F", flip(ProbF))

    print "Learned A:", learner.getQ("A")
    print "Learned B:", learner.getQ("B")
    print "Learned C:", learner.getQ("C")
    print "Learned D:", learner.getQ("D")
    print "Learned E:", learner.getQ("E")
    print "Learned F:", learner.getQ("F")

    print "CHOICE (A vs. B):", learner.choose("A","B")
    print "CHOICE (C vs. D):", learner.choose("C","D")
    print "CHOICE (E vs. F):", learner.choose("E","F")

# Optimistic model: ALPHA_POS > ALPHA_NEG
print "\nOPTIMISTIC MODEL"
test(0.4, 0.2, 0.1, 1000)

# Pessimistic model: ALPHA_POS < ALPHA_NEG
print "\nPESSIMISTIC MODEL"
test(0.2, 0.4, 0.1, 1000)

# Neutral model: ALPHA_POS = ALPHA_NEG
print "\nNEUTRAL MODEL"
test(0.3, 0.3, 0.1, 1000)
