import sys
import numpy as np
import cvxpy as cp
from numpy import ndarray

from policies.policy_abc import Policy


class OMD_Network(Policy):

	def __init__(self, capacity: int, catalog: int, time_window: int, users: int, caches: int) -> None:
		super().__init__(capacity, catalog, time_window)
		self.h = 1
		self.J = caches
		self.I = users
		self.x = np.full((self.J, self.N), self.k / self.N)  # The vector indicating which files are stored in cache
		self.R = 1 * self.I  # The number of files requested for each request
		self.z = np.zeros((self.I, self.J, self.N))
		self.learning_rate = self.calculate_lr()  # Learning rate of OMD
		self.init_problem()
		self.w = np.ones((self.I, self.J, self.N))
		self.P = 1

	def init_problem(self):
		self.a = cp.Parameter((self.J, self.N))  # Parameter projecting the x variable
		self.b = np.zer  # Parameter projecting the z variable
		self.a_var = cp.Variable((self.J, self.N), nonneg=True)
		self.b_var = cp.Variable((self.I, self.J, self.N), nonneg=True)
		self.constraints = [
			cp.sum(self.a_var, axis=1) <= self.k,
			cp.sum(self.b_var, axis=1) <= 1,
			self.b_var <= self.a_var,
			self.a_var <= 1
		]

		self.problem = cp.Problem(cp.Minimize(cp.sum_squares(self.b - self.b_var)), self.constraints)

	def get(self, y: ndarray) -> float:
		key = np.where(y == 1)[0][0]  # Todo change when multiple requests are made
		return self.x[key]

	def put(self, r_t: ndarray):
		z_hat_t_next = self.z * np.exp(self.learning_rate * r_t * self.w)
		x_t_next = z_t_next = self.project(z_hat_t_next)
		self.x = x_t_next
		self.z = z_t_next

	def calculate_lr(self):
		t1 = 2 * np.log(self.N / self.k)
		t2 = (max(abs(self.w)) ** 2) * (self.h ** 2) * self.T
		t3 = np.sqrt(t1 / t2)
		return t3

	def cost(self, r_t):
		total = 0
		for i in range(self.I):
			for j in range(self.J):
				for n in range(self.N):
					total += r_t[i, n] * self.z[i, j, n] * self.w[i, j, n]
		return total

	def cache_content(self):
		keys = np.arange(self.N)
		zipped = zip(keys, np.round(self.x, 6))
		return dict(zipped)

	def project(self, y):
		self.b.project_and_assign(y)
		self.problem.solve()
		if self.problem.status != "optimal":
			print("status:", self.problem.status)
		return self.a_var.value, self.b_var.value

	def return_x(self):
		return self.x
