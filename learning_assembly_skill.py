#!/home/wangchuang/anaconda3/envs/pytorchgpupy3.7/bin/python
import time
from math import pi
import os, sys, random
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

import rospy
from std_msgs.msg import String, Float64, Float64MultiArray
from environments.gazebo_env import envmodel

from DRL.SAC3_Pendulum import SAC3

from trajectory_planning.trajectory_planning_robot import trajectory_planning
from task_information.task_information import task_information

from IL.VMP.vmp import VMP
from icecream import ic

from torch.utils.tensorboard import SummaryWriter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train_model', type=str) # mode = 'evaluate_env','train_model' or 'test_model'
parser.add_argument('--load', default=False, type=bool) # load model True or False
parser.add_argument('--log_interval', default=20, type=int) # frequency to save model

parser.add_argument('--force_torque_max', default=[10,10,10,1,1,1], type=list)
parser.add_argument('--torque_max', default=1, type=float)
parser.add_argument('--motion_max', default=0.01, type=float)
parser.add_argument('--controller_arange', default=[25,25,100,100], type=list)
parser.add_argument('--trajectory_error', default=0.01, type=float)
parser.add_argument('--task', default='P_big', type=str)  # peg, L_peg, gear, gear_big, L_big, P_big, P_real, gear_real
parser.add_argument('--general_controller', default=False, type=bool) # load model True or False
parser.add_argument('--learning_method', default='SAC3', type=str)  # DDPG, SAC3, SAC4
parser.add_argument('--seed', default=1, type=int)
args = parser.parse_args()

state_dim = 18
action_dim = 6
max_action = 1
np.random.seed(args.seed)

directory = '/home/wangchuang/catkin_ws/src/wxm_assembly/wxm_skill_formalism/scripts/learning_result/test9-6/'   # ,6-4


def train(env, task, trajectory_planner, agent, Episode, Episode_step, Episode_reward):

    baseline = 'residual multimodal policy with attention'
    writer = SummaryWriter(directory+'train_data/reward')
    # reset environment
    random_init_pose = np.zeros(6)
    for m in range(6):
        random_init_pose[m] = task.start[m] + 1.5 * np.random.uniform(-task.errorspace[m], task.errorspace[m], size=1)

    estimated_goal_1 = np.zeros(6)
    for m in range(6):
        estimated_goal_1[m] = task.goal[m] + 0.02 * np.random.uniform(-task.errorspace[m], task.errorspace[m], size=1)

    estimated_goal = np.zeros(6)
    for m in range(6):
        estimated_goal[m] = estimated_goal_1[m] + 0.98 * np.random.uniform(-task.errorspace[m], task.errorspace[m], size=1)
    
    # load the trained policy for transfer learning
    #agent.load()
    
    env.refactor_environment('grasp_object')
    
    state = env.reset_env(start=task.start, goal=task.goal, estimated_start=random_init_pose, estimated_goal=estimated_goal, T=10)
    state, self_state_set, touch_state_set = env.input_initialization(state, 1, 32)

    # initialize the parameters for interaction, curriculum and learning
    max_episodes = 301
    max_steps = 120 + 50   # plus the move in free-space 20, 30, 50
    trajectory_steps = 80
    
    curriculum_episodes = 100
    curriculum_episodes2 = 200
    curriculum_error_level = 0.1
    curriculum_errorspace_level = 0
    
    gradient_steps = 200
    succeed_episodes = []
    cost_steps = []
    success_rates = []
    error_space = []
    error_level = []
    try:
        for i in range(max_episodes):
            # reset
            episode_reward = 0.0
            # initial point for control
            goal_command = env.tool_pos 
            action = np.zeros(6)

            # new episode

            # baselines: the activate point is far away from the pre-assembly point
            # MP1 for move to the generated activate point far away from the pre-assembly point
            init = np.zeros(6)
            error = [0.25, 0.25, 0, 0, 0, 0]  # 0.02, 0.05, 0.15, 0.25
            for m in range(6):
                init[m] = task.point_global_perception[m] + np.random.uniform(-error[m], error[m], size=1)
            waypoints = np.array([random_init_pose, init])
            trajectory = trajectory_planner.plan_trajectory(waypoints, 20)  # 3 seconds
            for t in range(len(trajectory)):
                env.step2(trajectory[t])
            # generate  far away from the activate point to pre-assembly point, then to goal point
            waypoints = np.array([trajectory[-1], random_init_pose])
            trajectory = trajectory_planner.plan_trajectory(waypoints, 50)  # 10, 20, 30, 50

            if trajectory_steps > 20:
                trajectory_steps = trajectory_steps - 1  # (150-i)/5 seconds
            waypoints = np.array([random_init_pose, estimated_goal])
            trajectory2 = trajectory_planner.plan_trajectory(waypoints, trajectory_steps)
            trajectory.extend(trajectory2)
            '''
            # ours: pre-assembly activate
            # increasing guidance speed from 80 steps (16s) to 15 steps (3s)
            if trajectory_steps > 20:
                trajectory_steps = trajectory_steps - 1  # (150-i)/5 seconds
            trajectory = trajectory_planner.generate_point(random_init_pose, estimated_goal, 0, 1, trajectory_steps)  # 3 seconds
            '''
            for t in range(max_steps):
                # agent interact with env
                action = agent.select_action(state[0], state[1], state[2])# SAC

                if t < len(trajectory):
                    goal_command = trajectory[t]
                else:
                    goal_command = estimated_goal
                
                next_state = env.step(action, goal_command)
                next_state, self_state_set, touch_state_set = env.resize_input(next_state, 1, 32, self_state_set, touch_state_set)
    
                reward, done, fail = env.get_reward(task.goal, goal_command)
    
                agent.replay_buffer.push(state[0], state[1], state[2], action, reward, next_state[0], next_state[1],  next_state[2], float(done))
    
                state = next_state
                episode_reward += reward
    
                if fail:
                    #break
                    goal_command = goal_command + 0.2*(task.start-goal_command)
                    env.step(action, goal_command) # -2*dx for action of pure RL to go back
                if done: # or fail:
                    succeed_episodes.append(i)
                    break
                
            success_num = 0
            success_rate = 0
            for j in range(len(succeed_episodes)):
                if (i-succeed_episodes[-(j+1)])<15:
                    success_num = success_num+1
                else:
                    success_rate = success_num/15
                    break
            print("Episode:", i, "Step:", t, "Total Reward:", episode_reward, "Success Rate:", success_rate, "Done:", done, "Fail:", fail)
            
            # adaptive curriculum
            # first stage for error
            if success_rate>0.7 and curriculum_error_level<1 and curriculum_errorspace_level<0.1:
                curriculum_error_level = curriculum_error_level + 0.05
            if success_rate<0.5 and curriculum_error_level>0.2 and curriculum_errorspace_level<0.01:
                curriculum_error_level = curriculum_error_level - 0.05
            # second stage for errorspace        
            if curriculum_error_level>0.98 and success_rate>0.7 and curriculum_errorspace_level<1:
                curriculum_errorspace_level = curriculum_errorspace_level+0.025
            if curriculum_error_level>0.98 and success_rate<0.5 and curriculum_errorspace_level>0.01:
                curriculum_errorspace_level = curriculum_errorspace_level-0.025
            task.errorspace = task.middle_errorspace + curriculum_errorspace_level * (task.final_errorspace - task.middle_errorspace)
            
            # reset environment
            random_init_pose = np.zeros(6)
            for m in range(6):
                random_init_pose[m] = task.start[m] + 1.5 * curriculum_error_level * np.random.uniform(-task.errorspace[m], task.errorspace[m],size=1)
            estimated_goal = np.zeros(6)
            for m in range(6):
                estimated_goal[m] = estimated_goal_1[m] + 0.98 * curriculum_error_level * np.random.uniform(-task.errorspace[m], task.errorspace[m], size=1)
            
            print(trajectory_steps, curriculum_error_level, task.errorspace, gradient_steps)
            
            if i < 10:
                reset_T=5
            else:
                reset_T=3
            
            # MP1 for regrasp and move to initialize the robot
            env.refactor_environment('disassembly')
            # MP1 for regrasp and move to initialize the robot
            env.refactor_environment('regrasp_object')
            
            state = env.reset_env(start=task.start, goal=task.goal, estimated_start=random_init_pose, estimated_goal=estimated_goal, T=reset_T)
            state, self_state_set, touch_state_set = env.input_initialization(state, 1, 32)
            
            env.assembly_state('4')
            # learning
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            if i > 20:
                print("learning")
                agent.update(gradient_steps)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            # saving model
            if i > 10 and i % args.log_interval == 0:
                agent.save()
    
            # storage episode data for visualization
            Episode.append(i)
            Episode_step.append(t)
            Episode_reward.append(episode_reward)
    
            writer.add_scalar('Learning_curves/episode_reward', episode_reward, global_step=i)
            writer.add_scalar('Learning_curves/episode_step', t, global_step=i)
            writer.add_scalar('Learning_curves/success_rate', success_rate, global_step=i)
            writer.add_scalar('Uncertaincy/error_level', curriculum_error_level, global_step=i)
            writer.add_scalar('Uncertaincy/errorspace_level', curriculum_errorspace_level, global_step=i)
            env.assembly_state('3')
            cost_steps.append(t)
            success_rates.append(success_rate)
            error_space.append(task.errorspace[0])
            error_level.append(curriculum_error_level)
        
        df1 = pd.DataFrame(data=np.array([cost_steps, success_rates, error_space, error_level]).T, columns=['cost_steps', 'success_rates', 'error_space', 'error_level'])
        df1.to_csv(directory + 'train_data/filename1.csv')
            
    except KeyboardInterrupt:
        rospy.signal_shutdown("KeyboardInterrupt")
        raise
        
    return Episode, Episode_step, Episode_reward


def control(env, task, trajectory_planner, agent):
    baseline = 'trajectory with random force exlporation'
    writer = SummaryWriter(directory+'unstructured_test_data-VMP-only/reward')
    # the robust errorspace learned by curriculum
    task.errorspace = np.array([0.01, 0.01, 0.01, 0.002, 0.002, 0.1])  # 0.15, 0.2, 0.25, 0.3
    
    print("ready assembly point for data augumentation")
    ra_point_init = np.array([-0.55, -0.55, 0.0000, 0.0000, 0.0000, 0.0000])
    ra_move_x = np.array([0.0500, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
    ra_move_y = np.array([0.0000, 0.0500, 0.0000, 0.0000, 0.0000, 0.0000])
    ra_points = []
    for i in range(4):
        for j in range(4):
            ra_points.append(ra_point_init+i*ra_move_x+j*ra_move_y)
    
    print("ready disturbance point for data augumentation")
    rd_point_init = np.array([[0.05, 0.0, 0.0000, 0.0000, 0.0000, 0.0000],[-0.05, 0.0, 0.0000, 0.0000, 0.0000, 0.0000],
                              [0.05, -0.15, 0.0000, 0.0000, 0.0000, 0.0000],[-0.05, -0.15, 0.0000, 0.0000, 0.0000, 0.0000]])

    env.refactor_environment('add_task', -ra_point_init+[0,0,0.002,0,0,0])
    n = 0
    #env.refactor_environment('add_disturbance', -ra_point_init+rd_point_init[n])

    # MP0 for grasp and move to initialize the robot
    env.refactor_environment('grasp_object')

    traj_files = ["/home/wangchuang/catkin_ws/src/wxm_assembly/wxm_robot_teleop/scripts/path_point_for_ILRRL5.csv"]

    trajs = np.array([np.loadtxt(f, delimiter=',') for f in traj_files])
    ic(trajs.shape)

    trajs2 = trajs[:, :102, :]
    ic(trajs2.shape)

    # via-point extraction and task-centric
    via_point = [0, 20, -1]
    off_set = [trajs[:, via_point[1], 1:7] - trajs[:, via_point[2], 7:13],
               trajs[:, via_point[2], 1:7] - trajs[:, via_point[2], 7:13]]

    # training
    vmp_set = []
    linear_traj_raw = trajs[:, 0, 1:7]
    for i in range(len(via_point) - 1):
        vmp = VMP(6, kernel_num=50, elementary_type='linear', use_out_of_range_kernel=False)
        temp_linear_traj_raw = vmp.train(trajs2[:, via_point[i]:via_point[i + 1], 0:7])
        vmp_set.append(vmp)
        linear_traj_raw = np.concatenate((linear_traj_raw, temp_linear_traj_raw), axis=0)

    try:
        max_episodes = 3
        max_steps = 120
        trajectory_steps = 50
        succeed_episodes = []
        cost_steps = []
        success_rates = []
        average_force = []


        for i in range(len(ra_points)-1):  #len(ra_points)
            # from fixture to workspace
            waypoints = np.array([task.start, task.point_global_perception])
            trajectory = trajectory_planner.plan_trajectory(waypoints, trajectory_steps)   # 3 seconds
            for t in range(len(trajectory)):
                env.step2(trajectory[t])

            # MP2 for pre-assembly
            '''
            ############## new episode #################
            # model-based method for task centric
            estimated_assembly_goal = env.goal + task.constraint_platform + task.offset_hole
            estimated_assembly_start = env.goal + task.constraint_platform + task.offset_hole + task.safe_distance
            placed_assembly_goal = ra_points[i] + task.constraint_platform + task.offset_hole
            print(estimated_assembly_start, estimated_assembly_goal, placed_assembly_goal)

            # trajectory planning and execution
            waypoints = np.array([task.point_global_perception, estimated_assembly_start])
            trajectory = trajectory_planner.plan_trajectory(waypoints, trajectory_steps)   # 3 seconds
            for t in range(len(trajectory)):
                env.step2(trajectory[t])
            '''

            # learning from demonstration for task centric
            # via_point modulation
            assembly_start = task.point_global_perception
            task_location = env.goal
            via_point = [assembly_start, off_set[0][0] + task_location, off_set[1][0] + task_location]
            # via_point = [start, trajs2[0, via_point[1], 1:4] + task, trajs2[0, via_point[2], 1:4] + task]
            ic(via_point,task_location)

            estimated_assembly_goal = via_point[2]
            estimated_assembly_start = via_point[1]
            placed_assembly_goal = ra_points[i] + task.constraint_platform + task.offset_hole
            print(estimated_assembly_start, estimated_assembly_goal, placed_assembly_goal)

            # reproduce (trajectory planning) and execution
            reproduced = trajs[:, 0, 0:7]
            linear_traj = trajs[:, 0, 1:7]
            for j in range(len(via_point) - 1):
                temp_reproduced, temp_linear_traj = vmp_set[j].roll(via_point[j], via_point[j + 1], 50)
                reproduced = np.concatenate((reproduced, temp_reproduced), axis=0)
                linear_traj = np.concatenate((linear_traj, temp_linear_traj), axis=0)

            #ic(reproduced.shape)
            #fig = plt.figure()
            #ax = fig.add_subplot(131, projection='3d')
            #ax.plot(trajs[0, :, 1], trajs[0, :, 2], trajs[0, :, 3], color='blue')
            #ax.plot(reproduced[:, 1], reproduced[:, 2], reproduced[:, 3], color="red")
            #ax.set_xlabel("x")
            #ax.set_ylabel("y")
            #ax.set_zlabel("z")
            #plt.show()

            for t in range(50):
                env.step2(reproduced[t, 1:7])

            # MP3 for assembly
            episode_reward = 0.0

            # generate the initial policy
            '''
            # model-based planning
            waypoints = np.array([estimated_assembly_start, estimated_assembly_goal])
            trajectory = trajectory_planner.plan_trajectory(waypoints, trajectory_steps)   # 3 seconds
            '''
            '''
            # visual servoing
            waypoints = np.array([task.point_global_perception, estimated_assembly_goal])
            trajectory = trajectory_planner.plan_trajectory(waypoints, 2*trajectory_steps)  # 3 seconds
            '''
            # VMP learning-based
            trajectory = reproduced[50:, 1:7]
            ic(trajectory.shape)

            step_contact_force = []
            # load the trained residual policy
            agent.load()

            env.assembly_state('3')  # assembly
            time.sleep(0.2)
            state = env.get_env2(estimated_assembly_goal)
            state, self_state_set, touch_state_set = env.input_initialization(state, 1, 32)

            for t in range(max_steps):
                # agent interact with env
                action = agent.select_action(state[0], state[1], state[2])# SAC
                # without residual policy
                action = np.array([0, 0, 0, 0, 0, 0])
                # static nominal policy by imitation learning or model-based planning
                if t < len(trajectory):
                    goal_command = trajectory[t]
                else:
                    goal_command = trajectory[-1]
                # dynamic nominal policy by visual servoing
                #goal_command = env.tool_pos + np.clip(0.8*(estimated_assembly_goal-env.tool_pos), -0.05, 0.05)
                print(estimated_assembly_goal,env.tool_pos,goal_command)

                next_state = env.step(action, goal_command)

                step_contact_force.append(next_state[1][0:3])

                next_state, self_state_set, touch_state_set = env.resize_input(next_state, 1, 32, self_state_set, touch_state_set)
        
                #estimated_assembly_goal = env.goal + task.constraint_platform + task.offset_hole
                reward, done, fail = env.get_reward(estimated_assembly_goal, goal_command)   #placed_assembly_goal
        
                state = next_state
                episode_reward += reward
        
                if done: # or fail:
                    succeed_episodes.append(i)
                    break
            
            success_rate = 0
            success_rate = len(succeed_episodes)/(i+1)
            
            print("Episode:", i, "Step:", t, "Total Reward:", episode_reward, "Success Rate:", success_rate, "Done:", done, "Fail:", fail)
        
            state = env.reset_env(start=estimated_assembly_start, goal=ra_points[i]+task.constraint_platform+task.offset_hole, estimated_start=estimated_assembly_start, estimated_goal=estimated_assembly_goal, T=10)
            state, self_state_set, touch_state_set = env.input_initialization(state, 1, 32)
            
            env.refactor_environment('set_state_task', -ra_points[i+1]+[0,0,0.002,0,0,0])
            #env.refactor_environment('set_state_disturbance', -ra_points[i+1]+rd_point_init[n%4])
            n = n+1
            
            # MP0 for regrasp and move to initialize the robot
            env.refactor_environment('regrasp_object')

            writer.add_scalar('Learning_curves/episode_reward', episode_reward, global_step=i)
            writer.add_scalar('Learning_curves/episode_step', t, global_step=i)
            writer.add_scalar('Learning_curves/success_rate', success_rate, global_step=i)
            
            cost_steps.append(t)
            success_rates.append(success_rate)
            average_force.append(np.abs(np.array(step_contact_force)).mean(axis=0))
        
        df1 = pd.DataFrame(data=np.array([cost_steps, success_rates, np.array(average_force)[:, 0], np.array(average_force)[:, 1], np.array(average_force)[:, 2]]).T,
                           columns=['cost_steps', 'success_rates', 'average_force_x', 'average_force_y', 'average_force_z'])
        df1.to_csv(directory + 'unstructured_test_data-VMP-only/filename1.csv')
         
    except KeyboardInterrupt:
        rospy.signal_shutdown("KeyboardInterrupt")
        raise


def main():
    try:
        agent = SAC3(state_dim, action_dim, max_action, directory)
    
        trajectory_planner = trajectory_planning()
    
        # define Episode varible to storage data
        Episode = []
        Episode_step = []
        Episode_reward = []
    
        env = envmodel()
        task = task_information()
        
        Episode, Episode_step, Episode_reward = train(env, task, trajectory_planner, agent, Episode, Episode_step, Episode_reward)
    
        # Episode, Episode_step, Episode_reward = test(env, task, trajectory_planner, agent, Episode, Episode_step, Episode_reward)
        
        # control(env, task, trajectory_planner, agent)

    except KeyboardInterrupt:
            rospy.signal_shutdown("KeyboardInterrupt")
            raise
            
if __name__ == '__main__':
    main()