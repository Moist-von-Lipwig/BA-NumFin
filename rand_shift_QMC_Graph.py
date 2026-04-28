import matplotlib.pyplot as plt
import time
from Minimal_Discrepancy_Sequences import rand_shift_MDS as rMDS

start_time =time.time()

H_d2 = rMDS.rand_Halton_Sequence(2, 1000)

print("--- %s seconds --- for 2 dimensions" % (time.time() - start_time))

start_time =time.time()

H_d30 = rMDS.rand_Halton_Sequence(30, 1000)

print("--- %s seconds --- for 30 dimensions" % (time.time() - start_time))

H_d2_x = H_d2[0, 1:]
H_d2_y = H_d2[1, 1:]

H_d30_x = H_d30[28, 1:]
H_d30_y = H_d30[29, 1:]

plt.scatter(H_d2_x, H_d2_y, 1, '0')
plt.show()

plt.scatter(H_d30_x, H_d30_y, 1, '0')
plt.show()