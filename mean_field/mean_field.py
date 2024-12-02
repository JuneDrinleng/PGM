import numpy as np
import matplotlib.pyplot as plt

# node weight and edge weight
alpha = np.array([0.1, 0.1])
beta = np.array([0.5, -0.3, -0.5])
weight = np.array([[-1, -1, -1], [0.2, 0.2, 0.2]])

# initial belief 
# consider it as Binomial Distribution
p_H = np.array([0.5, 0.5])
p_V = np.array([0.5, 0.5, 0.5])
# H or V =0

# history
p_H_history = [p_H.copy()]
p_V_history = [p_V.copy()]

for iteration in range(100):
    new_p_H = p_H.copy()
    new_p_V = p_V.copy()

    # update p_H
    for i in range(2):
        H_0 = 1
        H_1 = np.exp(
            alpha[i] * 1
            + weight[i][0] * p_V[0]
            + weight[i][1] * p_V[1]
            + weight[i][2] * p_V[2]
        )
        new_p_H[i] = H_1 / (H_0 + H_1)

    # update p_V
    for j in range(3):
        V_0 = 1 #V=0
        V_1 = np.exp(
            beta[j] * 1 + weight[0][j] * p_H[0] + weight[1][j] * p_H[1]
        )
        new_p_V[j] = V_1 / (V_0 + V_1)

    old_result = np.prod(np.concatenate((p_H, p_V)))
    new_result = np.prod(np.concatenate((new_p_H, new_p_V)))

    # save history
    p_H_history.append(1-new_p_H.copy())
    p_V_history.append(1-new_p_V.copy())

    # note the iteration number
    if abs(new_result - old_result) > 1e-6:
        stable = iteration+2

    p_H = new_p_H.copy()
    p_V = new_p_V.copy()


print(f"P(H_1=0): {1-p_H[0]}")
print(f"P(H_2=0): {1-p_H[1]}")
print(f"P(V_1=0): {1-p_V[0]}")
print(f"P(V_2=0): {1-p_V[1]}")
print(f"P(V_3=0): {1-p_V[2]}")
print(f"iterate to stable number : {stable}")

p_H_history = np.array(p_H_history)
p_V_history = np.array(p_V_history)



plt.plot(range(len(p_H_history)), p_H_history[:, 0], label="p(H_1=0)", marker="o")
plt.plot(range(len(p_H_history)), p_H_history[:, 1], label="p(H_2=0)", marker="o")
plt.plot(range(len(p_V_history)), p_V_history[:, 0], label="p(V_1=0)", marker="o")
plt.plot(range(len(p_V_history)), p_V_history[:, 1], label="p(V_2=0)", marker="o")
plt.plot(range(len(p_V_history)), p_V_history[:, 2], label="p(V_3=0)", marker="o")
plt.xlabel("iteration")
plt.ylabel("probability")
plt.legend()
plt.ylim(0, 1)
plt.savefig('T3.png')