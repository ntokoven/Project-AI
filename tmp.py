import numpy as np

W = np.random.randint(0, 10, (100, 10))
print(W)
x = np.ones(10)

y = W @ x
W_inv = np.linalg.pinv(W)
x_ = W_inv @ y

print(x_ - x)

