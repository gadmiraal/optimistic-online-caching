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
			rec_id = np.random.randint(0, self.N)
			while actual_id == rec_id:
				rec_id = np.random.randint(0, self.N)

			r_bar_t[rec_id] = 1
			return r_bar_t

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
		self.a.project_and_assign(y)
		self.problem.solve()
		if self.problem.status != "optimal":
			print("status:", self.problem.status)
		return self.a_var.value

	def return_x(self):
		return self.x

	def get_label(self) -> str:
		return self.name + " - " + str(self.chance*100) + "%"
