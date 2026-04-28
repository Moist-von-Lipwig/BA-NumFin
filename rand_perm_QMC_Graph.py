import matplotlib.pyplot as plt
import time
from Minimal_Discrepancy_Sequences.rand_Perm import (Rand_Perm_Faure_Sequence,
                                                     gen_rand_perm)

start_time =time.time()

Permutation = gen_rand_perm(31-1)
Fp31 = Rand_Perm_Faure_Sequence(31, 1000, 30, Permutation)

print("--- %s seconds --- for Prime 31 and 30 dimensions" % (time.time() - start_time))

start_time =time.time()

Permutation = gen_rand_perm(2-1)
Fp2 = Rand_Perm_Faure_Sequence(2, 1000, 2, Permutation)

print("--- %s seconds --- for Prime 2 and 2 dimensions" % (time.time() - start_time))

F_p2_1x = Fp2[0, 1:]
F_p2_2y = Fp2[1, 1:]

F_p31_1x = Fp31[0, 1:]
F_p31_2y = Fp31[1, 1:]

F_p31_29x = Fp31[28, 1:]
F_p31_30y = Fp31[29, 1:]

plt.scatter(F_p2_1x, F_p2_2y, 1, '0')
plt.show()

plt.scatter(F_p31_1x, F_p31_2y, 1, '0')
plt.show()

plt.scatter(F_p31_29x, F_p31_30y, 1, '0')
plt.show()