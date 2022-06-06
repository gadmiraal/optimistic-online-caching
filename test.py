import numpy as np

m = 3
Q = np.array([[-0.01, 0.0095, 0.0005, 0], [0.08, -0.1, 0.02, 0], [0, 0, -1, 1], [0, 0, 0, 0]])
init_pop = np.array([1, 0, 0, 0])
T = Q[0:m, 0:m]
I = np.identity(m)
U_1 = np.linalg.matrix_power(-T, -1)
U_2 = np.linalg.matrix_power(-T, -2)
alpha_zero = init_pop[m]
s_zero = Q[0:m, m]
alpha = init_pop[0:m]
ones = np.ones(m)
s = 0
mu = alpha @ U_1 @ ones
sigma = 2 * alpha @ U_2 @ ones
print(mu)
print(sigma)

t1 = alpha_zero + alpha @ np.linalg.inv(s * I - T) @ s_zero
print(t1)

pi = alpha @ U_1 * (mu ** -1)
print(pi)