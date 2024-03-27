from typing import Union

import numpy as np
import matplotlib.pyplot as plt

from icecream import ic

pinv_rcond = 1.4e-08

class VMP:
    def __init__(self, dim, kernel_num=30, kernel_std=0.1, elementary_type='linear', use_out_of_range_kernel=True):
        self.kernel_num = kernel_num
        if use_out_of_range_kernel:
            self.centers = np.linspace(1.2, -0.2, kernel_num)  # (K, )
        else:
            self.centers = np.linspace(1, 0, kernel_num)  # (K, )

        self.kernel_variance = kernel_std ** 2
        self.var_reci = - 0.5 / self.kernel_variance
        self.elementary_type = elementary_type
        self.lamb = 0.01
        self.dim = dim
        self.n_samples = 100
        self.kernel_weights = np.zeros(shape=(kernel_num, self.dim))

        self.h_params = None
        self.y0 = None
        self.g = None


    def __psi__(self, can_value: Union[float, np.ndarray]):
        """
        compute the contribution of each kernel given a canonical value
        """
        
        return np.exp(np.square(can_value - self.centers) * self.var_reci)


    def __Psi__(self, can_values: np.ndarray):
        """
        compute the contributions of each kernel at each time step as a (T, K) matrix, where
        can_value denotes the sampled canonical values, and is a (T, ) array, where
        T is the total number of time steps.
        """
        return self.__psi__(can_values[:, None])


    def h(self, x):
        if self.elementary_type == 'linear':
            return np.matmul(self.h_params, np.matrix([[1], [x]]))
        else:
            return np.matmul(self.h_params, np.matrix([
                [1], [x], [np.power(x, 2)], [np.power(x, 3)], [np.power(x, 4)], [np.power(x, 5)]
                ]))


    def linear_traj(self, can_values: np.ndarray):
        """
        compute the linear trajectory (T, dim) given canonical values (T, )
        """
        if self.elementary_type == 'linear':
            can_values_aug = np.stack([np.ones(can_values.shape[0]), can_values])
        else:
            can_values_aug = np.stack([np.ones(can_values.shape[0]), can_values, 
                                       np.power(can_values, 2), np.power(can_values, 3), 
                                       np.power(can_values, 4), np.power(can_values, 5)])
            
        return np.einsum("ij,ik->kj", self.h_params, can_values_aug)  # (n, 2) (T, 2)

    

    def via_point(self, data):
        raw_v = self.differential(data[:, 1:4])
        v_avg = self.moving_average(raw_v, 4)
        
        raw_a = self.differential(v_avg)
        via_index = np.argmax(raw_a)
        
        print(via_index, 
              np.linalg.norm(data[via_index, 1:4]-data[-1, 1:4]), 
              [data[0, 1:7], data[via_index, 1:7], data[-1, 1:7]])
    
        return v_avg, raw_a, via_index
    
    
    def moving_average(raw_v, w):
        return np.convolve(raw_v, np.ones(w), "valid") / w
    
    
    def exponential_moving_average(raw_v):
        v_ema = []
        v_pre = 0
        beta = 0.9
        for i, t in enumerate(raw_v):
            v_t = beta * v_pre + (1 - beta) * t
            v_ema.append(v_t)
            v_pre = v_t
    
        v_ema_corr = []
        for i, t in enumerate(v_ema):
            v_ema_corr.append(t / (1 - np.power(beta, i+1)))
    
        return v_ema_corr
    
    
    def differential(data):
        raw_v = []
        
        for i in range(0, len(data)-1):
            tem_v = np.linalg.norm(data[i+1] - data[i])
            raw_v.append(tem_v)
        
        return(raw_v)
    
    
    def get_weights(self):
        return self.kernel_weights


    def get_flatten_weights(self):
        return self.kernel_weights.flatten('F')


    def set_weights(self, ws: np.ndarray):
        """
        set weights to VMP

        Args:
            ws: (kernel_num, dim)
        """
        if np.shape(ws)[-1] == self.dim * self.kernel_num:
            self.kernel_weights = np.reshape(ws, (self.kernel_num, self.dim), 'F')
        elif np.shape(ws)[0] == self.kernel_num and np.shape(ws)[-1] == self.dim:
            self.kernel_weights = ws
        else:
            raise Exception(f"The weights have wrong shape. "
                            f"It should have {self.kernel_num} rows (for kernel number) "
                            f"and {self.dim} columns (for dimensions), but given is {ws.shape}.")

    
    def save_weights_to_file(self, filename):
        np.savetxt(filename, self.kernel_weights, delimiter=',')


    def load_weights_from_file(self, filename):
        self.kernel_weights = np.loadtxt(filename, delimiter=',')
        
        
    def get_position(self, t):
        x = 1 - t
        return np.matmul(self.__psi__(x), self.kernel_weights)

    def get_target(self, t):
        action = np.transpose(self.h(1-t)) + self.get_position(t)
        return action
    
    
    def set_start(self, y0):
        self.y0 = y0
        self.h_params = np.stack([self.g, self.y0 - self.g])


    def set_goal(self, g):
        self.g = g
        self.h_params = np.stack([self.g, self.y0 - self.g])


    def set_start_goal(self, y0, g):
        self.set_start(self, y0)
        self.set_goal(self, g)


    def train(self, trajectories):
         """
         Assume trajectories are regularly sampled time-sequences.
         """
         if len(trajectories.shape) == 2: # (n, T, 2)
             trajectories = np.expand_dims(trajectories, 0)
    
         n_demo, self.n_samples, self.dim = trajectories.shape
         self.dim -= 1
    
         can_value_array = self.can_sys(1, 0, self.n_samples)  # canonical variable (T)
         Psi = self.__Psi__(can_value_array)  # (T, K) squared exponential (SE) kernels
    
         if self.elementary_type == 'linear':
             y0 = trajectories[:, 0, 1:].mean(axis=0)
             g = trajectories[:, -1, 1:].mean(axis=0)
             self.h_params = np.stack([g, y0-g])
         else:
             # min_jerk
             y0 = trajectories[:, 0:3, 1:].mean(axis=0)
             dy0 = (y0[1, 2:] - y0[0, 2:]) / (y0[1, 1] - y0[0, 1])
             dy1 = (y0[2, 2:] - y0[1, 2:]) / (y0[2, 1] - y0[1, 1])
             ddy0 = (dy1 - dy0) / (y0[1, 1] - y0[0, 1])
             
             g = trajectories[:, -2:, 1:].mean(axis=0)
             dg0 = (g[1, 2:] - g[0, 2:]) / (g[1, 1] - g[0, 1])
             dg1 = (g[2, 2:] - g[1, 2:]) / (g[2, 1] - g[1, 1])
             ddg = (dg1 - dg0) / (g[1, 1] - g[0, 1])
    
             self.h_params = self.get_min_jerk_params(y0, g, dy0, dg1, ddy0, ddg)
    
         self.y0 = y0
         self.g = g
         linear_traj = self.linear_traj(can_value_array)  # (T, dim)  elementary trajectory
         shape_traj = trajectories[..., 1:] - np.expand_dims(linear_traj, 0)  # (N, T, dim) - (1, T, dim) shape modulation
    
         pseudo_inv = np.linalg.pinv(Psi.T.dot(Psi), pinv_rcond)  # (K, K)
         self.kernel_weights = np.einsum("ij,njd->nid", pseudo_inv.dot(Psi.T), shape_traj).mean(axis=0)  #
         ic(Psi.shape, shape_traj.shape, self.kernel_weights.shape)
         
         return linear_traj
 
    
    def roll(self, y0, g, n_samples=None):
        """
        reproduce the trajectory given start point y0 (dim, ) and end point g (dim, ), return traj (n_samples, dim)
        """
        n_samples = n_samples or self.n_samples
        can_values = self.can_sys(1, 0, n_samples)  # canonical variable (T)

        if self.elementary_type == "minjerk":
            dv = np.zeros(y0.shape)
            self.h_params = self.get_min_jerk_params(y0, g, dv, dv, dv, dv)
        else:
            self.h_params = np.stack([g, y0 - g])

        linear_traj = self.linear_traj(can_values) # (T, dim)  elementary trajectory

        psi = self.__Psi__(can_values)  # (T, K) squared exponential (SE) kernels
        ic(psi.shape, self.kernel_weights.shape)
        traj = linear_traj + np.einsum("ij,jk->ik", psi, self.kernel_weights)

        time_stamp = 1 - np.expand_dims(can_values, 1)
        
        return np.concatenate([time_stamp, traj], axis=1), linear_traj


    @staticmethod
    def can_sys(t0, t1, n_sample):
        """
        return the sampled values of linear decay canonical system

        Args:
            t0: start time point
            t1: end time point
            n_sample: number of samples
        """
        return np.linspace(t0, t1, n_sample)


    @staticmethod
    def get_min_jerk_params(y0, g, dy0, dg, ddy0, ddg):
        b = np.stack([y0, dy0, ddy0, 
                      g, dg, ddg])
        A = np.array([[1, 1, 1, 1, 1, 1], [0, 1, 2, 3, 4, 5], [0, 0, 2, 6, 12, 20], 
                      [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0]])

        return np.linalg.solve(A, b)
     
    
    
def scale_and_reproduce(trajs, vmp_set, via_t, off_set, target_offset):
    start = trajs[:, 0, 1:4][0]
    task = trajs[:, -1, 7:10][0] + np.array(target_offset)
    via_point = [start, off_set[0][0] + task, off_set[1][0] + task]
    ic(via_point)

    # reproduce
    scaled_VMP_p003 = trajs[:, 0, 0:4]
    linear_traj = trajs[:, 0, 1:4]
    for i in range(len(via_point)-1):
        temp_reproduced, temp_linear_traj = vmp_set[i].roll(via_point[i], via_point[i+1], via_t[i+1]-via_t[i])
        ic(temp_reproduced, temp_linear_traj)
        # planned trajectory is directly used as the base trajectory in transfer phase with index col for alignment
        if i < 1:
            temp_reproduced = np.insert(temp_linear_traj, 0, np.linspace(0, 1, temp_linear_traj.shape[0]), axis=1)
        scaled_VMP_p003 = np.concatenate((scaled_VMP_p003, temp_reproduced), axis=0)
        linear_traj = np.concatenate((linear_traj, temp_linear_traj), axis=0)
        
    return scaled_VMP_p003, linear_traj


def draw(trajs, linear_traj, scaled_VMP_p, scaled_VMP_n):
    trajs2 = trajs[:, :98, :]
    
    fig = plt.figure(dpi=300)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.6)
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.legend(ncol=2, loc="upper right")
    ax1.plot(trajs[0, :, 1], trajs[0, :, 2], trajs[0, :, 3], color='blue', label='Demonstration')
    ax1.plot(scaled_VMP_p003[:, 1], scaled_VMP_p003[:, 2], scaled_VMP_p003[:, 3], color="red", label='VMP')
    ax1.plot(scaled_VMP_n003[:, 1], scaled_VMP_n003[:, 2], scaled_VMP_n003[:, 3], color="red")

    t = np.linspace(0, 1, trajs2[0, :, 1].shape[0])

    ax2.legend(ncol=1, loc="upper right")
    
    ax2.plot(t, trajs2[0, :, 1], color="b", linestyle="-", label='Demonstration')
    ax2.plot(t, linear_traj[:, 0], color="r", linestyle="-.", label='VMP_h(x)')
    ax2.plot(t, scaled_VMP_n003[:, 1], color="r", linestyle="-", alpha=0.5, label='VMP')
    
    plt.savefig('visualize_IL_real_data.png', dpi=300)
    plt.show()
    
    

if __name__ == '__main__':
    ################################ test with real data ############################
    traj_files = ["./path_point_for_ILRRL1.csv"]

    trajs = np.array([np.loadtxt(f, delimiter=',') for f in traj_files])
    ic(trajs.shape)

    trajs2 = trajs[:, :98, :]
    ic(trajs2.shape)

    ######### via-point #########
    # via-point extraction and task-centric
    via_t = [0, 23, trajs2[0, :, 1].shape[0]-1]
    off_set = [trajs[:, via_t[1], 1:4] - trajs[:, via_t[2], 7:10], 
               trajs[:, via_t[2], 1:4] - trajs[:, via_t[2], 7:10]]

    # training
    vmp_set = []
    linear_traj_raw = trajs[:, 0, 1:4]
    for i in range(len(via_t)-1):
        vmp = VMP(3, kernel_num=int(0.5*via_t[i+1]), elementary_type='linear', use_out_of_range_kernel=False)
        temp_linear_traj_raw = vmp.train(trajs2[:, via_t[i]:via_t[i+1], 0:4])
        vmp_set.append(vmp)
        linear_traj_raw = np.concatenate((linear_traj_raw, temp_linear_traj_raw), axis=0)
        

    ########################### test 1 ###########################
    # scale to variable position [0.03, 0.03, 0]
    # via_point modulation
    offset1 = np.array([0.03, 0.03, 0])
    scaled_VMP_p003, linear_traj = scale_and_reproduce(trajs, vmp_set, via_t, off_set, offset1)


    ########################### test 2 ###########################
    # scale to variable position [-0.03, -0.03, 0]
    # via_point modulation
    offset1 = np.array([-0.03, -0.03, 0])
    scaled_VMP_n003, linear_traj = scale_and_reproduce(trajs, vmp_set, via_t, off_set, offset1)
    
    draw(linear_traj, trajs)