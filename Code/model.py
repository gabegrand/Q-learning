import math
import random

# General constants for the model
POSITIVE = 1
NEGATIVE = 0

# Main QLearner class
class QLearner(object):
    def __init__(self, pos, neg, beta):
        self.Q = {}
        self.ALPHA_POS = pos
        self.ALPHA_NEG = neg
        self.BETA = beta

    def getQ(self, state):
        return self.Q.get(state, 0)

    def learn(self, state, real_reward):
        Q = self.getQ(state)
        delta = real_reward - Q
        if delta > 0:
            self.Q[state] = Q + self.ALPHA_POS * delta
        else:
            self.Q[state] = Q + self.ALPHA_NEG * delta

    # We use a softmax to determine the best option
    def choose(self, stimA, stimB):
        Q_A = float(self.getQ(stimA))
        Q_B = float(self.getQ(stimB))

        # Equation 2 from paper
        probChooseA = (math.exp(Q_A/self.BETA) /
                      (math.exp(Q_A/self.BETA) + math.exp(Q_B/self.BETA)))

        if random.random() < probChooseA:
            choice = stimA
        else:
            choice = stimB

        return choice
