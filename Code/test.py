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
def train(pos, neg, beta, trials):
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

    return learner

def simple_test(learner):
    print "CHOICE (A vs. B):", learner.choose("A","B")
    print "CHOICE (C vs. D):", learner.choose("C","D")
    print "CHOICE (E vs. F):", learner.choose("E","F")

def test(learner, trials):
    chosen_A = 0
    chosen_B = 0
    for i in xrange(trials):
        choice = learner.choose("A","B")
        if choice == "A": chosen_A += 1
        else: chosen_B += 1

    print "PERCENT CHOSE A:", float(chosen_A)/trials
    print "PERCENT CHOSE B:", float(chosen_B)/trials

def chooseA_test(learner, trials):
    # Set of more neutral stimuli
    others = ["C", "D", "E", "F"]

    chosen_A = 0
    chosen_other = 0

    for i in xrange(trials):
        choice = learner.choose("A",random.choice(others))
        if choice == "A": chosen_A += 1
        else: chosen_other += 1

    return float(chosen_A)/trials

def avoidB_test(learner, trials):
    # Set of more neutral stimuli
    others = ["C", "D", "E", "F"]

    chosen_B = 0
    chosen_other = 0

    for i in xrange(trials):
        choice = learner.choose("B",random.choice(others))
        if choice == "B": chosen_B += 1
        else: chosen_other += 1

    return float(chosen_other)/trials

# Varying ALPHA_POS and ALPHA_NEG:
optimist = train(0.4, 0.2, 0.2, 1000)
print "\nOPTIMISTIC MODEL"
print "Percent Chose A:", chooseA_test(optimist, 100)
print "Percent Avoided B:", avoidB_test(optimist, 100)

pessimist = train(0.2, 0.4, 0.2, 1000)
print "\nPESSIMISTIC MODEL"
print "Percent Chose A:", chooseA_test(pessimist, 100)
print "Percent Avoided B:", avoidB_test(pessimist, 100)

neutral = train(0.3, 0.3, 0.2, 1000)
print "\nNEUTRAL MODEL"
print "Percent Chose A:", chooseA_test(neutral, 100)
print "Percent Avoided B:", avoidB_test(neutral, 100)

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
