import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from traces.adversarial import Adversarial
from traces.poisson_point import PoissonPoint
from traces.fixed_pop import FixedPop
from traces.sliding_pop import SlidingPop

# Todo batch request?

configs = [
	"configs/system_config_lru.json",
	"configs/system_config_omd.json",
	"configs/system_config_oomd_00.json",
	"configs/system_config_oomd_70.json",
	"configs/system_config_oomd_100.json"
]

names = [
	"LRU",
	"OMDne",
	"OOMD: 0%",
	"OOMD: 70%",
	"OOMD: 100%",
]

caches = []
for config in configs:
	env = Environment(config)
	N = env.N
	T = env.T
	trace = Adversarial(N, T)

	env.set_trace(trace)

	cs = env.caches
	env.execute()

	for i, cache in enumerate(cs):
		cache.pretty_print(i)
		caches.append(cache)

for i, cache in enumerate(caches):
	T = len(cache.cost)
	avg = np.zeros(T)
	avg[0] = cache.cost[0]
	for t in range(1, T):
		avg[t] = sum(cache.cost[:t]) / t

	# print(cache.policy.__class__.__name__)
	# if cache.policy.__class__.__name__ == "OOMD":
	# 	print(cache.policy.chance)
	# 	print(names[i])
	#
	# print("Avg: " + str(avg[T-1]))
	# print("=====================================")

	plt.plot(np.arange(T), avg, "--", label=names[i])

plt.title("Average cache cost for trace: " + trace.get_name())
plt.xlabel("Time")
plt.ylabel("Average cache cost")
plt.legend(loc="best")
plt.show()



# print("Misses_1: " + str(sum(omd_cache_1.misses))) # + ", Hits_1: " + str(omd_cache_2.hits))
# print("Misses_2: " + str(sum(omd_cache_2.misses))) # + ", Hits_2: " + str(omd_cache_2.hits))
# hit_ratio_1 = omd_cache_1.get_avg_hit_ratio()
# hit_ratio_2 = omd_cache_2.get_avg_hit_ratio()
#
# plt.plot(np.arange(hit_ratio_2.shape[0]), hit_ratio_2, ":")



# division = np.random.randint(1, 10)
# roll = np.random.randint(1, int(N/division))
#
# trace, trace_name = sliding_popularity(T, N, roll, division)
# # trace, trace_name = adversarial_trace_3(T, N)
# # tace, trace_name = poisson_shot_noise(T, N)
# # trace, trace_name = fixed_popularity(T, N)
# T = trace.shape[0]
#
# LRUcache = LRU(k, N)
# LFUcache = LFU(k, N)
# OMDcache = OMD(k, N, T)
# OptimalCache = Optimal(k, N)
#
# caches = [LRUcache, LFUcache, OMDcache, OptimalCache]
# caches_names = []
# for cache in caches:
#     caches_names.append(type(cache).__name__)
#
# df = pd.DataFrame(columns=caches_names)
#
# def do(cache: Cache, trace):
#     T = len(trace)
#     typee = type(cache).__name__
#     misses = []
#
#     begin = time.perf_counter()
#     for x in trace:
#         y = cache.get(x)
#         misses.append(1.0 - y)
#         cache.put(x)
#     end = time.perf_counter()
#
#     summed = np.zeros(T)
#     summed[0] = misses[0]
#     for i in range(1, T):
#         summed[i] = sum(misses[:i]) / i
#
#     elapsed_time = round(end - begin, 6)
#     total_misses = sum(misses)
#     string = typee + " misses: " + str(abs(round(total_misses, 6))) + ", time: " + str(elapsed_time) #+ " and final cache: " + str(cache.cache_content())
#     return string, summed, elapsed_time
#
# def do_optimal():
#     misses = np.zeros(trace.shape[0])
#     for i in range(trace.shape[0]):
#         y = cache.get(trace[i])
#         misses[i] = (1.0 - y)
#
#     summed = np.zeros(trace.shape[0])
#     summed[0] = misses[0]
#     for i in range(1, T):
#         summed[i] = sum(misses[:i]) / i
#
#     total_misses = sum(misses)
#     string = "Optimal misses: " + str(abs(round(total_misses, 6)))
#
#     return string, summed
#
# def do_all(caches, caches_names, trace, trace_name, dataframe):
#     strings = []
#     misses = []
#     times = []
#     for i in range(len(caches)-1):
#         string, miss, elapsed_time = do(caches[i], trace)
#         strings.append(string)
#         misses.append(miss)
#         times.append(elapsed_time)
#
#     static_cache = OMDcache.return_x()
#     print(static_cache)
#     OptimalCache.set_cache(static_cache)
#     optimal_string, optimal_misses = do_optimal()
#     strings.append(optimal_string)
#     misses.append(optimal_misses)
#
#     t_time = np.arange(trace.shape[0])
#     for i in range(len(misses)):
#         miss = misses[i]
#         name = caches_names[i]
#         plt.plot(t_time, miss, label=name)
#
#     # plt.plot(t_time, optimal_misses, label="Optimal static")
#
#     plt.title("Average cache misses over time for trace: " + trace_name)
#     plt.xlabel("Time")
#     plt.ylabel("Average cache misses")
#     plt.legend(loc="upper right")
#     plt.show()
#
#     for string in strings:
#         print(string)
#
#     # zipped = list(zip(misses, times))
#     # df = dataframe.append(zipped)
#
#     # writer = pd.ExcelWriter('data.xlsx', engine='xlsxwriter')
#     # df.to_excel(writer, sheet_name='Advesarial3')
#     # writer.save()
#
#
# do_all(caches, caches_names, trace, trace_name, df)

