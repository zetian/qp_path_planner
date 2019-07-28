import osqp
import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse as sparse
from scipy.linalg import block_diag
import time

class FrenetPathPlanner:
    def __init__(self):
        self.horizon = 10
        self.uniform_ds = 0.1
        self.ds = np.ones(self.horizon - 1)*self.uniform_ds
        self.init_ddl = 0.0
        self.init_dl = 0.0
        self.init_l = 0.0
        self.l_ref = np.zeros(self.horizon)
        self.l_weight = 600
        self.dl_weight = 120
        self.ddl_weight = 30000
        self.dddl_weight = 100
        self.l_max = np.ones((self.horizon, 1))
        self.l_min = np.ones((self.horizon, 1))
        self.ddl_max = np.ones((self.horizon, 1))
        self.ddl_min = np.ones((self.horizon, 1))
        self.l_res = np.zeros(self.horizon)
    
    def set_reference_l(self, l_ref):
        self.l_ref = l_ref
    
    def set_initial_condition(self, init_l, init_dl, init_ddl):
        self.init_ddl = init_ddl
        self.init_dl = init_dl
        self.init_l = init_l

    def set_horizon(self, horizon):
        self.horizon = horizon

    def set_l_boundary(self, l_max, l_min):
        self.l_max = np.reshape(l_max, (self.horizon, 1))
        self.l_min = np.reshape(l_min, (self.horizon, 1))

    def set_ddl_boundary(self, ddl_max, ddl_min):
        self.ddl_max = np.ones((self.horizon, 1))*ddl_max
        self.ddl_min = np.ones((self.horizon, 1))*ddl_min
    
    def set_uniform_ds(self, ds):
        self.uniform_ds = ds
        self.ds = np.ones(self.horizon - 1)*self.uniform_ds
    
    def set_ds(self, ds):
        self.ds = ds

    def compute_P_q(self):
        weight = np.array([[self.l_weight, 0.0, 0.0, 0.0],
                    [0.0, self.dl_weight, 0.0, 0.0],
                    [0.0, 0.0, self.ddl_weight, 0.0],
                    [0.0, 0.0, 0.0, self.dddl_weight]])
        P = sparse.kron(sparse.eye(self.horizon), weight)
        q = np.zeros(self.horizon*4)
        for i in range(self.horizon):
            q[i*4] = -self.l_weight*self.l_ref[i]
        return P, q
    

    def compute_A(self):
        Ad = []
        for i in range(self.horizon - 1):
            f_i = np.array([[1.0, self.ds[i], 0.0, 0.0],
                            [0.0, 1.0, self.ds[i], 0.0],
                            [0.0, 0.0,  1.0, self.ds[i]]])  
            Ad.append(f_i)  
        Ax = sparse.csr_matrix(block_diag(*Ad))
        f_2 = np.array([[-1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0]])
        Ay = sparse.kron(sparse.eye(self.horizon - 1), f_2)
        off_set = np.zeros(((self.horizon - 1)*3, 4))
        Ax = sparse.hstack([Ax, off_set])
        Ay = sparse.hstack([off_set, Ay])
        Aeq = Ax + Ay
        ineq_l = np.array([1.0, 0.0, 0.0, 0.0])
        ineq_ddl = np.array([0.0, 0.0, 1.0, 0.0])

        Aineq_l = sparse.kron(sparse.eye(self.horizon), ineq_l)
        Aineq_ddl = sparse.kron(sparse.eye(self.horizon), ineq_ddl)


        A_init_l = np.zeros(self.horizon*4)
        A_init_l[0] = 1
        A_init_dl = np.zeros(self.horizon*4)
        A_init_dl[1] = 1

        A = sparse.vstack([Aeq, Aineq_l, Aineq_ddl, A_init_l, A_init_dl]).tocsc()
        return A

    def compute_u_l(self):
        ueq = np.zeros(((self.horizon - 1)*3, 1))
        leq = ueq
        uineq = np.vstack([self.l_max, self.ddl_max])
        lineq = np.vstack([self.l_min, self.ddl_min])
        u_init = np.array([[self.init_l],[self.init_dl]])
        l_init = u_init
        l = np.vstack([leq, lineq, l_init])
        u = np.vstack([ueq, uineq, u_init])
        return u, l

    def __call__(self):
        P, q = self.compute_P_q()
        A = self.compute_A()
        u, l = self.compute_u_l()
        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, warm_start=True, verbose=True)
        res = prob.solve()
        for i in range(self.horizon):
            self.l_res = res.x[i*4]
    
    def plot(self):
        plt.plot(self.l_res)
        plt.plot(self.l_min)
        plt.plot(self.l_max)
        plt.show()

if __name__ == '__main__':
    horizon = 400
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
    
    planner = FrenetPathPlanner()
    # planner.set_horizon(horizon)
    # planner.set_l_boundary(l_max, l_min)
    planner()
    planner.plot()