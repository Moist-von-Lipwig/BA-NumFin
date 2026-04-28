import time
import matplotlib.pyplot as plt
import numpy.random as npr

start_time =time.time()

PR = npr.uniform(0,1, (1000, 1000))

print("--- %s seconds ---" % (time.time() - start_time))

PR_x = PR[0, 1:]
PR_y = PR[1, 1:]

plt.scatter(PR_x, PR_y, 1, '0')
plt.show()