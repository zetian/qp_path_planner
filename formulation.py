import osqp
import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse as sparse
from scipy.linalg import block_diag
import time

horizon = 10
dt = 0.1

n_states = 4

ref_l = 0.0
ref_v = 0.0
ref_a = 0.0


init_ddl = 0.0
init_dl = 0.0
init_l = 0.0

ref_l = 0.0

acc_max = 0.2
acc_min = -0.2

pos_weight = 600
jerk_weight = 100
acc_weight = 30000
vel_weight = 120

weight = np.array([[pos_weight, 0.0, 0.0, 0.0],
                    [0.0, vel_weight, 0.0, 0.0],
                    [0.0, 0.0, acc_weight, 0.0],
                    [0.0, 0.0, 0.0, jerk_weight]])
weight_q = np.array([-pos_weight*ref_l, -vel_weight*ref_v, -acc_weight*ref_a, 0.0])

P = sparse.kron(sparse.eye(horizon), weight)
q = np.kron(np.ones(horizon), weight_q)
print("P: ", P.shape)
print("q: ", q.shape)




l_max = []
l_min = []
for i in range(horizon):
    u = 0.2
    if (i > 150 and i < 160):
        u = -0.1
    l_max.append(u)
    # l_max = np.append(l_max, u)
    l = -0.2
    if (i > 40 and i < 50):
        l = 0.1
    # l_min = np.append(l_min, u)
    l_min.append(l)
l_max = np.reshape(l_max, (horizon, 1))
l_min = np.reshape(l_min, (horizon, 1))

a_max = np.ones((horizon, 1))*acc_max
a_min = np.ones((horizon, 1))*acc_min


f_1 = np.array([[1.0, dt, 0.0, 0.0],
                [0.0, 1.0, dt, 0.0],
                [0.0, 0.0,  1.0, dt]])
f_2 = np.array([[-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0]])



off_set = np.zeros(((horizon - 1)*(n_states - 1), n_states))
Ax = sparse.kron(sparse.eye(horizon - 1), f_1)
Ay = sparse.kron(sparse.eye(horizon - 1), f_2)
Ax = sparse.hstack([Ax, off_set])
Ay = sparse.hstack([off_set, Ay])
# print("Ax: ", Ax.shape)
# print("Ay: ", Ay.shape)
Aeq = Ax + Ay

# print(Aeq)

ineq_l = np.array([1.0, 0.0, 0.0, 0.0])
ineq_a = np.array([0.0, 0.0, 1.0, 0.0])

Aineq_l = sparse.kron(sparse.eye(horizon), ineq_l)
Aineq_a = sparse.kron(sparse.eye(horizon), ineq_a)


A_init_l = np.zeros(horizon*n_states)
A_init_l[0] = 1
A_init_v = np.zeros(horizon*n_states)
A_init_v[1] = 1

# A = sparse.vstack([Aeq, Aineq]).tocsc()
A = sparse.vstack([Aeq, Aineq_l, Aineq_a, A_init_l, A_init_v]).tocsc()

ueq = np.zeros(((horizon - 1)*(n_states - 1), 1))
# ueq = np.reshape(ueq, ((horizon - 1)*(n_states - 1), 1))

leq = ueq

# uineq = l_max
# lineq = l_min

uineq = np.vstack([l_max, a_max])
lineq = np.vstack([l_min, a_min])

# print("leq: ", leq.shape)
# print("lineq: ", lineq.shape)

u_init = np.array([[init_l],[init_dl]])
l_init = u_init

l = np.vstack([leq, lineq, l_init])
u = np.vstack([ueq, uineq, u_init])

print("P: ", P)
print("q: ", q)
print("A: ", A)
print("u: ", u)
print("l: ", l)


start = time.time()
prob = osqp.OSQP()
prob.setup(P, q, A, l, u, warm_start=True, verbose=True)
res = prob.solve()

end = time.time()
print("Computation time: ", end - start)

x = []
for i in range(horizon):
    x.append(res.x[i*4])

plt.plot(x)
plt.plot(l_min)
plt.plot(l_max)
plt.show()

