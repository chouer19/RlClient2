#!/usr/bin/env python
from __future__ import print_function

from __future__ import print_function

import argparse
import logging
import random
import time
import math

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

import tensorflow as tf
import cv2
import sys
#sys.path.append("game/")
#import wrapped_flappy_car as game
import wrapped_carla_sim as simulator
import random
import numpy as np
import threading
from utils import proPrint
from collections import deque
sys.path.append('controller/')
from controller import *

GAME = 'angry-car' # the name of the game being played for log files
#ACTIONS = 2 # number of valid actions
ACTIONS = 20 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.16 # starting value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon
#INITIAL_EPSILON = 0.2 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
FPS = 20

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 10, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([4, 4, 64, 128])
    b_conv3 = bias_variable([128])

    W_conv4 = weight_variable([3, 3, 128, 256])
    b_conv4 = bias_variable([256])

    W_conv5 = weight_variable([3, 3, 256, 256])
    b_conv5 = bias_variable([256])

    W_fc1 = weight_variable([2304, 256])
    b_fc1 = bias_variable([256])

    W_fc2 = weight_variable([256, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    #W_fc3 = weight_variable([256, ACTIONS])
    #b_fc3 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 160, 160, 10])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 2) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 2) + b_conv4)
    #h_pool4 = max_pool_2x2(h_conv4)

    h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, 1) + b_conv5)
    #h_pool4 = max_pool_2x2(h_conv4)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv5_flat = tf.reshape(h_conv5, [-1, 2304])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)
    #h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def send_control(game):
    while True:
        game.send_control()
        #time.sleep(0.02)

def trainNetwork(s, readout, h_fc1, sess, args, con):
    with make_carla_client(args.host, args.port) as client:
        game = simulator.CarlaGame(client, args)

        # define the cost function
        a = tf.placeholder("float", [None, ACTIONS])
        y = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y - readout_action))
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

        game.initialize_game()
        con.state = game.state
        #con.state = game.get_state()
        #tSend = threading.Thread(target=send_control, args = (game,))
        #tSend.setDaemon(True)
        #tSend.start()

        # store the previous observations in replay memory
        D = deque()

        # get the first state by doing nothing and preprocess the image to 80x80x4
        do_nothing = np.zeros(ACTIONS)
        direction_index = 0
        angle1_index = 2
        angle2_index = 4
        angle4_index = 6
        angle8_index = 8
        angle16_index = 10
        angle32_index = 12
        angle64_index = 14
        angle128_index = 16
        angle256_index = 18

        do_nothing[direction_index] = 1
        do_nothing[angle1_index] = 1
        do_nothing[angle2_index] = 1
        do_nothing[angle4_index] = 1
        do_nothing[angle8_index] = 1
        do_nothing[angle16_index] = 1
        do_nothing[angle32_index] = 1
        do_nothing[angle64_index] = 1
        do_nothing[angle128_index] = 1
        do_nothing[angle256_index] = 1
        game.set_steer(0)
        x_t, r_0, terminal = game.frame_step()
        x_t = cv2.cvtColor(cv2.resize(x_t, (160, 160)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
        s_t = np.stack((x_t, x_t, x_t, x_t, x_t, x_t, x_t, x_t, x_t, x_t), axis=2)

        # saving and loading networks
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        epsilon = INITIAL_EPSILON
        t = 0
        while "flappy car" != "angry car":
            ## choose an action epsilon greedily
            readout_t = readout.eval(feed_dict={s : [s_t]})[0]
            ##print(readout_t)
            a_t = np.zeros([ACTIONS])
            ##
            action_index = 0
            direction_index = 0
            angle1_index = 2
            angle2_index = 4
            angle4_index = 6
            angle8_index = 8
            angle16_index = 10
            angle32_index = 12
            angle64_index = 14
            angle128_index = 16
            angle256_index = 18

            # scale down epsilon
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            if t % FRAME_PER_ACTION == 0:
                random_value = random.random()
                if random_value <= epsilon:
                    random_value /= epsilon
                    #max_random_action_index = ( 1 / (random_value ** 2 + 1) - 1/2 )  * 2 * 640 - 60
                    max_random_action_index = int(random_value ** 2  * 620)
                    action_index = random.randint(0, max(0,min(511,max_random_action_index)) )
                    steer = action_index
                    direction_index += random.randint(0,1)
                    if direction_index == 0:
                        action_index *= -0.001
                    else:
                        action_index *= 0.001

                    a_t[direction_index] = 1
                    a_t[angle1_index + steer % 2] = 1
                    steer /= 2
                    a_t[angle2_index + steer % 2] = 1
                    steer /= 2
                    a_t[angle4_index + steer % 2] = 1
                    steer /= 2
                    a_t[angle8_index + steer % 2] = 1
                    steer /= 2
                    a_t[angle16_index + steer % 2] = 1
                    steer /= 2
                    a_t[angle32_index + steer % 2] = 1
                    steer /= 2
                    a_t[angle64_index + steer % 2] = 1
                    steer /= 2
                    a_t[angle128_index + steer % 2] = 1
                    steer /= 2
                    a_t[angle256_index + steer % 2] = 1
                    proPrint.prCyan("\t\t\t\t----------Random Action----------")
                    #proPrint.prCyan("\t       {:.3f}".format(action_index))
                    proPrint.prCyan("\t\t\t{:.3f}".format(action_index))
                else:
                    direction_index = np.argmax(readout_t[0:2])
                    angle1_index = np.argmax(readout_t[2:4]) + 2
                    angle2_index = np.argmax(readout_t[4:6]) + 4
                    angle4_index = np.argmax(readout_t[6:8]) + 6
                    angle8_index = np.argmax(readout_t[8:10]) + 8
                    angle16_index = np.argmax(readout_t[10:12]) + 10
                    angle32_index = np.argmax(readout_t[12:14]) + 12
                    angle64_index = np.argmax(readout_t[14:16]) + 14
                    angle128_index = np.argmax(readout_t[16:18]) + 16
                    angle256_index = np.argmax(readout_t[18:20]) + 18

                    a_t[direction_index] = 1
                    a_t[angle1_index] = 1
                    a_t[angle2_index] = 1
                    a_t[angle4_index] = 1
                    a_t[angle8_index] = 1
                    a_t[angle16_index] = 1
                    a_t[angle32_index] = 1
                    a_t[angle64_index] = 1
                    a_t[angle128_index] = 1
                    a_t[angle256_index] = 1
                    action_index = round( (-1 * a_t[0] + a_t[1]) * 0.001 * \
                                          (a_t[3] + a_t[5]*2 + a_t[7]*4 + a_t[9] * 8 + a_t[11]*16 + \
                                           a_t[13]*32 + a_t[15]*64 + a_t[17]*128 + a_t[19]*256 ), 3)
                    #if t%10 == 0:
                    #    proPrint.prOrange(readout_t)
            else:
                pass
                #a_t[1] = 1 # do nothing
            # run the selected action and observe next state and reward
            game.set_steer(action_index)
            x_t1_colored, r_t, terminal = game.frame_step()
            if t % 5 ==0 or terminal:
            #if True:
                states = game.get_states()
                message = '{time:,}\t'
                message += 'epsilon:{episilon:.4f}\t'
                message += 'action:{steer:+.3f}\t'
                message += 'reward:{reward:+.3f}\t'
                message += 'Dis:{dis:.2f}\t'
                message += 'Yaw:{yaw:.2f}\t'
                message += 'OffRoad:{offroad:.0%}\t'
                message = message.format(
                    time = t,
                    steer=action_index,
                    reward=r_t,
                    episilon=epsilon,
                    dis=states[1],
                    yaw=states[2],
                    offroad=states[4])
                if math.fabs(states[2]) < 1 and states[1] < 0.5:
                    proPrint.prGreen(message)
                elif math.fabs(states[2]) < 2 and states[1] < 1:
                    proPrint.prBlue(message)
                elif math.fabs(states[2]) < 5 and states[1] < 2.5:
                    proPrint.prYellow(message)
                else:
                    proPrint.prRed(message)

            if terminal:
                game.new_game()

            x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (160, 160)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (160, 160, 1))
            s_t1 = np.append(x_t1, s_t[:, :, :9], axis=2)

            # store the transition in D
            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            # only train if done observing
            if t > OBSERVE:
                # sample a minibatch to train on
                minibatch = random.sample(D, BATCH)
                # get the batch variables
                s_j_batch = [d[0] for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]

                y_batch = []
                readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
                for i in range(0, len(minibatch)):
                    terminal = minibatch[i][4]
                    # if terminal, only equals reward
                    if terminal:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + GAMMA *  
                                       ( np.max(readout_j1_batch[i][0:2] ) + \
                                         np.max(readout_j1_batch[i][2:4] ) + \
                                         np.max(readout_j1_batch[i][4:6] ) + \
                                         np.max(readout_j1_batch[i][6:8] ) + \
                                         np.max(readout_j1_batch[i][8:10] ) + \
                                         np.max(readout_j1_batch[i][10:12]) +\
                                         np.max(readout_j1_batch[i][12:14]) +\
                                         np.max(readout_j1_batch[i][14:16]) +\
                                         np.max(readout_j1_batch[i][16:18]) +\
                                         np.max(readout_j1_batch[i][18:20])
                                      )  )
                # perform gradient step
                train_step.run(feed_dict = {
                    y : y_batch,
                    a : a_batch,
                    s : s_j_batch}
                )
            # update the old values
            s_t = s_t1
            t += 1

            # save progress every 10000 iterations
            if t % 10000 == 0:
                saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

            # print info
            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"


def playGame(args):
    contr = Controller()
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess,args,contr)


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-r', '--road',
        metavar='R',
        default='waypoints',
        help='road location of waypoints road')
    argparser.add_argument(
        '-rl', '--road_length',
        metavar='RL',
        default=15,
        type=int,
        help='length of stright roads')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-m', '--map-name',
        metavar='M',
        default='Town01',
        help='plot the map of the current city (needs to match active map in '
             'server, options: Town01 or Town02)')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    playGame(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
