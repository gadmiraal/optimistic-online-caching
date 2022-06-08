import sys
import numpy as np
import cvxpy as cp
from numpy import ndarray

from policies.policy_abc import Policy


class OOMD(Policy):

	def __init__(self, capacity: int, catalog: int, time_window: int, chance=0.0) -> None:
		super().__init__(capacity, catalog, time_window)
		self.x = np.full(self.N, self.k / self.N)  # The vector indicating which files are stored in cache
		self.y = np.full(self.N, self.k / self.N)  # The vector storing the proxy action
		self.h = 1  # Todo
		self.R = 1  # The number of files requested for each request
		self.learning_rate = self.calculate_lr()  # Learning rate of OMDne
		self.init_problem()
		self.P = 1
		self.chance = chance
		self.name = "OOMD"
		self.r_t_next = None

	def init_problem(self):
		self.a = cp.Parameter(self.N)
		self.a_var = cp.Variable(self.N, nonneg=True)
		self.constraints = [
			cp.sum(self.a_var) <= self.k,
			self.a_var <= 1
		]
		self.problem = cp.Problem(cp.Minimize(cp.sum_squares(self.a - self.a_var)), self.constraints)

	def get(self, y: ndarray) -> float:
		key = np.where(y == 1)[0][0]  # Todo change when multiple requests are made
		return self.x[key]

	def put(self, r_t: ndarray):
		y_hat_t_next = self.y * np.exp(self.learning_rate * r_t * self.w)
		y_t_next = self.project(y_hat_t_next)
		r_bar_t_next = self.make_recommendation()
		x_hat_t_next = y_t_next * np.exp(self.learning_rate * r_bar_t_next * self.w)
		x_t_next = self.project(x_hat_t_next)
		self.x = x_t_next
		self.y = y_t_next
		# print("r_t: " + str(np.where(r_t == 1)[0][0]))
		# print("r_bar_t_next: " + str(np.where(r_bar_t_next == 1)[0][0]))

	# x_t_next_2 = self.project2(y_t_next)
	# x_t_next_3 = self.project3(y_t_next)
	# sum_1 = sum(x_t_next)
	# sum_2 = sum(x_t_next_2)
	# sum_3 = sum(x_t_next_3)

	def set_future_request(self, r_t_next: ndarray):
		if r_t_next is not None:
			self.r_t_next = r_t_next


	def make_recommendation(self) -> ndarray:
		roll = np.random.random()
		if roll <= self.chance:
			return self.r_t_next
		else:
			actual_id = np.where(self.r_t_next == 1)[0][0]
			r_bar_t = np.zeros(self.N)
			file_id = np.random.randint(0, self.N)
			while actual_id != file_id:
				file_id = np.random.randint(0, self.N)
			r_bar_t[file_id] = 1
			return r_bar_t

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
		self.a.project_and_assign(y)
		self.problem.solve()
		if self.problem.status != "optimal":
			print("status:", self.problem.status)
		return self.a_var.value

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
