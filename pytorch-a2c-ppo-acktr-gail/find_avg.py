import numpy as np
x = "0.40 0.39 0.36 0.39 0.44 0.44 0.38 0.41 0.41 0.46 0.39 0.33"
s = x.split(" ")
# assert len(s) == 12
y = [float(i) for i in s]
print(np.mean(y))
# print(np.mean(y) * 100 / 257)
