import numpy as np

training_scores = np.load('training_scores.npy')

last_100_scores_average = np.mean(training_scores[-100:])

print(last_100_scores_average)