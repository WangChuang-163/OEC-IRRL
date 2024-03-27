import rospy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def via_point(data):
    raw_v = differential(data[:, 1:4])
    v_avg = moving_average(raw_v, 4)
    
    raw_a = differential(v_avg)
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



def path_reading(dir):
    data = pd.read_csv(dir)
    print(np.array(data))
    
    return np.array(data)


def data_plot(data):
    fig = plt.figure(num=1, dpi=300)
    
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    plt.subplots_adjust(hspace=0.3)

    ax1.cla()
    ax2.cla()
    ax3.cla()
    
    t = range(0, len(data))
    ax1.set_xlabel('s')
    ax1.set_ylabel('Position(m)')
    ax1.legend(ncol=3,loc="upper right")
    ax1.set_ylim(-0.5, 0.5)
    ax1.plot(t, data[:, 1], label='tool_x')
    ax1.plot(t, data[:, 2], label='tool_y')
    ax1.plot(t, data[:, 3], label='tool_z')
    
    v, a, via_index = via_point(data)
    
    t = range(0, np.size(v))
    ax2.set_xlabel('s')
    ax2.set_ylabel('Velocity')
    ax2.plot(t, v, label='velocity')
    
    t = range(0, np.size(a))
    ax3.set_xlabel('s')
    ax3.set_ylabel('Acceleration')
    ax3.plot(t, a, label='acceleration')

    plt.savefig('visualize_demonstration.pdf', dpi=300)
    plt.show()
    

def main():
    try:
        data = path_reading("./path_point_for_ILRRL8-peg.csv")
        data_plot(data)
    except KeyboardInterrupt:
        rospy.signal_shutdown("KeyboardInterrupt")
        raise

if __name__ == '__main__': main()