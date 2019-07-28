import osqp
import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse as sparse
from scipy.linalg import block_diag
import time

horizon = 400
dt = 0.1

init_ddl = 0
init_dl = 0
init_l = 0.0

ref_l = 0

jerk_max = 5
jerk_min = -5

acc_max = 0.2
acc_min = -0.2

l_max = []
l_min = []
for i in range(horizon):
    u = 0.2
    if (i > 150 and i < 160):
        u = -0.1
    l_max.append(u)
    l = -0.2
    if (i > 40 and i < 50):
        l = 0.1
    l_min.append(l)
l_max = np.reshape(l_max, (horizon, 1))
l_min = np.reshape(l_min, (horizon, 1))

j_max = np.ones((horizon, 1))*jerk_max
j_min = np.ones((horizon, 1))*jerk_min

a_max = np.ones((horizon, 1))*acc_max
a_min = np.ones((horizon, 1))*acc_min
# a_max[0] = np.inf
# a_min[0] = -np.inf


a0 = init_ddl*np.ones((horizon, 1))
# print(a0)
v0 = init_dl*np.ones((horizon, 1))
l0 = init_l*np.ones((horizon, 1))
L_ref = ref_l*np.ones((horizon, 1))

H = np.tril(np.ones((horizon, horizon))*dt, -1)
# print(np.dot(H, a0))

H2 = np.dot(H, H)
H3 = np.dot(np.dot(H, H), H)
pos_weight = 600
jerk_weight = 100
acc_weight = 30000
vel_weight = 120

Q = np.eye(horizon)*pos_weight
R = np.eye(horizon)*jerk_weight
G = np.eye(horizon)*acc_weight
F = np.eye(horizon)*vel_weight

C0 = np.dot(H2, a0) + np.dot(H, v0) + l0



# print("C0: ", C0)
# print("l_max: ", l_max - C0)
P = np.dot(np.dot(np.transpose(H3), Q), H3) + R
P = P + np.dot(np.dot(np.transpose(H2), F), H2)
P = P + np.dot(np.dot(np.transpose(H), G), H)
# print(P)
P = sparse.csc_matrix(P)

q = np.dot(np.dot(np.transpose(C0 - L_ref), Q), H3)
q = q + np.dot(np.dot(np.transpose(v0 + np.dot(H, a0)), F), H2)
q = q + np.dot(np.dot(np.transpose(a0), G), H)
q = np.reshape(q, (horizon, 1))
# print("q:", q)
# A = np.vstack([np.eye(horizon), H, H3])
A = np.vstack([H, H3])
# print(A)
A = sparse.csc_matrix(A)

l_max = np.reshape(l_max, (horizon, 1)) - C0
l_min = np.reshape(l_min, (horizon, 1)) - C0
# l_max[0] = np.inf
# l_max[1] = np.inf
# l_max[2] = np.inf
# print("l_min: ", l_min)
# l_min[0] = -np.inf
# l_min[1] = -np.inf
# l_min[2] = -np.inf

# u = np.vstack([j_max, a_max, l_max])
# l = np.vstack([j_min, a_min, l_min])

u = np.vstack([a_max, l_max])
l = np.vstack([a_min, l_min])
# print("l: ", l)
# print("u: ", u)
# print(l_min)

start = time.time()
prob = osqp.OSQP()

print("P: ", P)
print("q: ", q)
print("A: ", A)
print("u: ", u)
print("l: ", l)

prob.setup(P, q, A, l, u, warm_start=True, verbose=True)
res = prob.solve()

end = time.time()
print("Computation time: ", end - start)

# print(res.x)
res_jerk = res.x
# print(C0)
x = np.dot(H3, res_jerk) + np.reshape(C0, (1, horizon))
# print(x)

plt.plot(x[0])
plt.plot(l_min)
plt.plot(l_max)
plt.show()



# A = sparse.vstack([sparse.csr_matrix(np.eye(horizon)), sparse.csr_matrix(H), sparse.csr_matrix(H3)]).tocsc()
# print(A)
# print("P: ", P)
# print("q: ", q)



# print(np.transpose(H))
# print("Q: ", Q)
# print("R: ", R)

# print(np.dot(np.dot(H, H), H))