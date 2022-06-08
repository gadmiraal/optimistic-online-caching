import sys
import numpy as np
import cvxpy as cp
from numpy import ndarray

from policies.policy_abc import Policy


class OMD(Policy):

	def __init__(self, capacity: int, catalog: int, time_window: int) -> None:
		super().__init__(capacity, catalog, time_window)
		self.x = np.full(self.N, self.k / self.N)  # The vector indicating which files are stored in cache
		self.q = 2  # Todo also change gradient_mirror_map
		self.p = 2  # Todo
		self.h = 1  # Todo
		self.R = 1  # The number of files requested for each request
		self.learning_rate = self.calculate_lr()  # Learning rate of OMD
		self.init_problem()
		self.P = 1
		self.name = "OMD"

	def init_problem(self):
		self.x_par = cp.Parameter((self.N, ))
		self.x_var = cp.Variable(self.N, nonneg=True)
		self.constraints = [
			cp.sum(self.x_var) <= self.k,
			self.x_var <= 1
		]

		self.problem = cp.Problem(cp.Minimize(cp.sum_squares(self.x_par - self.x_var)), self.constraints)

	def get(self, y: ndarray) -> float:
		key = np.where(y == 1)[0][0]  # Todo change when multiple requests are made
		return self.x[key]

	def put(self, r_t: ndarray):
		x_hat_t_next = self.x * np.exp(self.learning_rate * r_t * self.w)
		x_t_next = self.project(x_hat_t_next)
		self.x = x_t_next

	# x_t_next_2 = self.project2(y_t_next)
	# x_t_next_3 = self.project3(y_t_next)
	# sum_1 = sum(x_t_next)
	# sum_2 = sum(x_t_next_2)
	# sum_3 = sum(x_t_next_3)

	def calculate_lr(self):
		# t1 = (self.q - 1) * (self.k ** 2)
		# t2 = (self.k ** (-2 / self.p)) - (self.N ** (-2 / self.p))
		# t3 = (max(abs(self.w)) ** 2) * (self.h ** 2)
		# t4 = ((self.R / self.h) ** (2 / self.p)) * self.T
		# return np.sqrt((t1 * t2) / (t3 * t4))

		t1 = 2 * np.log(self.N / self.k)
		t2 = (max(abs(self.w)) ** 2) * (self.h ** 2) * self.T
		t3 = np.sqrt(t1 / t2)
		return t3

	# return 1 * np.sqrt(2 * self.k / self.T)

	def cost(self, r_t):
		return np.sum(self.w * r_t * (1 - self.x))

	def cache_content(self):
		keys = np.arange(self.N)
		zipped = zip(keys, np.round(self.x, 6))
		return dict(zipped)

	def project(self, y):
		self.x_par.project_and_assign(y)
		self.problem.solve()
		if self.problem.status != "optimal":
			print("status:", self.problem.status)
		return self.x_var.value

	def project2(self, y):
		d = np.argsort(y)
		y_man = sum(abs(y))

		for b in np.arange(self.N - 1, self.N - 1 - self.k, -1):
			m_b = (self.k + b - self.N + 1) / (y_man - sum(y[d[b + 1:]] * self.P))
			if (b < self.N - 1 and y[d[b]] * m_b * self.P < 1 <= y[d[b + 1]] * m_b * self.P) or \
					(b == self.N - 1 and y[d[b]] * m_b * self.P < 1 <= sys.maxsize * m_b * self.P):
				for i in np.arange(b + 1, self.N, 1):
					y[d[i]] = 1 / (m_b * self.P)
				self.P = m_b * self.P
				y = y * self.P
				s = sum(y)
				print(s)
				return y

		s = sum(y)
		print(s)

	def project3(self, y):
		z_i = np.flip(np.argsort(y))
		z = y[z_i]

		x = np.zeros(self.N)

		M1 = np.asarray([])
		M2 = np.where(z != 0)[0]
		M3 = np.where(z == 0)[0]

		x = self.helper(M1, M2, M3, x, y, z)

		if x[0] > 1:
			M1 = np.asarray([0])
			M2 = np.where(z != 0)[0]
			M2 = np.delete(M2, np.where(M2 == 0))
			M3 = np.where(z == 0)[0]

			x = self.helper(M1, M2, M3, x, y, z)

		if np.where(x > 1)[0].size > 1:
			print(x)
			raise Exception()

		return x

	def helper(self, M1, M2, M3, x, y, z):
		while True:
			rho = 2 * (M1.size - self.k + sum(z[M2])) / M2.size

			if M1.size > 0:
				x[M1] = 1
			if M2.size > 0:
				x[M2] = z[M2] - (rho / 2)
			if M3.size > 0:
				x[M3] = 0

			S = np.where(x < 0)[0]
			M2 = M2[~np.isin(M2, S)]
			M3 = np.unique(np.append(M3, S)).astype(int)
			if S.size == 0:
				return x

	def return_x(self):
		return self.x

	def get_label(self) -> str:
		return self.name
