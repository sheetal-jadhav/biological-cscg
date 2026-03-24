

import numpy as np
import matplotlib.pyplot as plt

P = 9           
PATTERN_NAMES = ['A','B','C','D','delay','a','b','c','d']

# sequences: AB delay ab, CD delay cd
seq1 = np.array([0, 1, 4, 5, 6])  # A B delay a b
seq2 = np.array([2, 3, 4, 7, 8])  # C D delay c d

# Random sparse patterns: each pattern has 'each_input' active inputs at 100 Hz
each_input = 10
x_rate = np.zeros((P, N))
for p_each in range(P):
    active = np.random.choice(N, size=each_input, replace=False)
    x_rate[p_each, active] = input_fr