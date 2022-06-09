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

	def get(self, r_t: ndarray) -> float:
		key = np.where(r_t == 1)[0][0]  # Todo change when multiple requests are made
		return self.x[key]

	def put(self, r_t: ndarray):
		x_hat_t_next = self.x * np.exp(self.learning_rate * r_t * self.w)
		x_t_next = self.project(x_hat_t_next)
		self.x = x_t_next

	def calculate_lr(self):
		t1 = 2 * np.log(self.N / self.k)
		t2 = (max(abs(self.w)) ** 2) * (self.h ** 2) * self.T
		t3 = np.sqrt(t1 / t2)
		return t3

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

	def return_x(self):
		return self.x

	def get_label(self) -> str:
		return self.name
