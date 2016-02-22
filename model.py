# General constants for the model
POSITIVE = 1
NEGATIVE = 0

# Learning rates for positive and negative
ALPHA_POS = 0.4
ALPHA_NEG = 0.2

# Main QLearner class
class QLearner(object):
    def __init__(self):
        self.Q = {}

    def getQ(self, state):
        return self.Q.get(state, 0)

    def learn(self, state, real_reward):
        Q = self.getQ(state)
        delta = real_reward - Q
        if delta > 0:
            self.Q[state] = Q + ALPHA_POS * delta
        else:
            self.Q[state] = Q + ALPHA_NEG * delta

# We use a softmax to determine the best option
