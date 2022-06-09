import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.hstack import hstack
from numpy import ndarray

from policies.policy_abc import Policy


class OOMD_Network(Policy):

	def __init__(self, capacity: int, catalog: int, time_window: int, chance: float, users: int, caches: int) -> None:
		super().__init__(capacity, catalog, time_window)
		self.h = 1
		self.I = users
		self.J = caches
		self.x = np.full((self.J, self.N), self.k / self.N)  # The vector indicating which files are stored in cache
		self.R = 1 * self.I  # The number of files requested for each request
		self.z = np.full((self.I, self.J, self.N), min(1 / self.J, self.k / self.N))
		self.y = np.full((self.I, self.J, self.N), min(1 / self.J, self.k / self.N))  # The vector storing the proxy action
		self.learning_rate = self.calculate_lr()  # Learning rate of OMD
		self.init_problem()
		self.w = np.ones((self.I, self.J, self.N))
		self.P = 1
		self.chance = chance
		self.r_t_next = None
		self.name = "OOMD Network"

	def init_problem(self):
		self.x_par = cp.Parameter((self.J, self.N))  # Parameter projecting the x variable
		self.z_par = {}  # Parameter projecting the z variable
		for j in range(self.J):
			self.z_par[j] = cp.Parameter((self.I, self.N))

		self.x_var = cp.Variable((self.J, self.N), nonneg=True)
		self.z_var = {}
		for j in range(self.J):
			self.z_var[j] = cp.Variable((self.I, self.N), nonneg=True)

		constraints = [cp.sum(self.x_var, axis=1) <= self.k, self.x_var <= 1,
						[cp.sum(self.z_var[j]) <= 1 for j in range(self.J)],
						[self.z_var[j] <= self.x_var for j in range(self.J)]]

		constraints = self.flatten(constraints)

		objective = []
		for j in range(self.J):
			objective.append(cp.sum_squares(self.z_par[j] - self.z_var[j]))

		self.problem = cp.Problem(
			cp.Minimize(
				cp.sum_squares(hstack(objective))
			), constraints)

	def flatten(self, a):
		if type(a) is list:
			return [element for item in a for element in self.flatten(item)]
		else:
			return [a]

	def get(self, y: ndarray) -> float:
		key = np.where(y == 1)[0][0]  # Todo change when multiple requests are made
		return self.x[key]

	def put(self, r_t: ndarray):
		# z_hat_t_next = self.z * np.exp(self.learning_rate * r_t * self.w)
		# x_t_next, z_t_next = self.project(z_hat_t_next)
		# self.x = x_t_next
		# self.z = z_t_next

		y_hat_t_next = self.y * np.exp(self.learning_rate * r_t * self.w)
		_, y_t_next = self.project(y_hat_t_next)
		r_bar_t_next = self.make_recommendation()
		z_hat_t_next = y_t_next * np.exp(self.learning_rate * r_bar_t_next * self.w)
		x_t_next, z_t_next = self.project(z_hat_t_next)
		self.x = x_t_next
		self.y = y_t_next
		self.z = z_t_next


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
		total = 0
		for i in range(self.I):
			for j in range(self.J):
				for n in range(self.N):
					total += r_t[i, n] * (1 - self.z[i, j, n]) * self.w[i, j, n]
		return total

	def cache_content(self):
		keys = np.arange(self.N)
		zipped = zip(keys, np.round(self.x, 6))
		return dict(zipped)

	def project(self, y):
		for j in range(self.J):
			self.z_par[j].project_and_assign(y[:, j, :])
		self.problem.solve()
		if self.problem.status != "optimal":
			print("status:", self.problem.status)

		z_var = np.zeros((self.I, self.J, self.N))
		for j in range(self.J):
			z_var[:, j, :] = self.z_var[j].value

		return self.x_var.value, z_var

	def return_x(self):
		return self.x

	def utility(self, r_t):
		total = 0
		for i in range(self.I):
			for j in range(self.J):
				for n in range(self.N):
					total += r_t[i, n] * self.z[i, j, n] * self.w[i, j, n]
		return total

	def get_label(self) -> str:
		return self.name + " - " + str(self.chance * 100) + "%"
