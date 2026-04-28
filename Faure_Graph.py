import matplotlib.pyplot as plt
import time
from Minimal_Discrepancy_Sequences.Faure import Faure_Sequence

start_time =time.time()

Fp31 = Faure_Sequence(31, 1000, 30)

print("--- %s seconds --- for Prime 31 and 30 dimensions" % (time.time() - start_time))

start_time =time.time()

Fp2 = Faure_Sequence(2, 1000, 2)

print("--- %s seconds --- for Prime 2 and 2 dimensions" % (time.time() - start_time))

F_p2_1x = Fp2[0, 1:]
F_p2_2y = Fp2[1, 1:]

F_p3_1x = Fp31[0, 1:]
F_p3_2y = Fp31[1, 1:]

F_p3_29x = Fp31[28, 1:]
F_p3_30y = Fp31[29, 1:]

plt.scatter(F_p2_1x, F_p2_2y, 1, '0')
plt.show()

plt.scatter(F_p3_1x, F_p3_2y, 1, '0')
plt.show()

plt.scatter(F_p3_29x, F_p3_30y, 1, '0')
plt.show()