##################################################################
#####################   Author: Tianshi xie   ####################
#####################       09/20/2017        ####################
##################################################################

from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("../Game/")
from Game import MyGame
import random
import numpy as np
from collections import deque
import pygame, sys
import matplotlib.pyplot as plt

GAME = 'plane'
ACTIONS = 3
GAMMA = 0.99
VIEW = 500
ADVENTURE = 3000000
FINAL_RANDOM_THRESHOLD = 0.0001
ROBOT_MEM = 500000
BATCH = 32
ACTION_FRAME = 1
cost_his = []

MODEL_PLAY  = 0
MODEL_TRAINING = 1

# tensorboard --logdir=/Users/xietianshi/PycharmProjects/xts_mac/Network
# tensorboard --logdir=C:\xts\logs
# tensorboard --logdir=C:\Users\18713\Desktop\Class\machine learning\Project\Submition for final\xts_mac_10_16_midterm_submit\Network\logs
# localhost:6006

class Neuro_Network:
    def __init__(self):
        self.model = MODEL_TRAINING  #0 : PLAY, 1 : TRAINING
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) #tf.InteractiveSession()
        # To create a game
        self.game = MyGame()
        # To create a memory
        self.Memory = deque()
        self.t = 0

        self.actions = np.zeros([ACTIONS])
        self.action_index = 0

        if self.model == MODEL_TRAINING:
            self.initial_random_threshold = 0.1
        elif self.model == MODEL_PLAY:
            self.initial_random_threshold = 0.0001

        self.random_threshold = self.initial_random_threshold

        self.total_reward = 0
        self.avg_reward = 0
        self.ep_reward = 0
        self.ep_rewards = []
        self.avg_loss = 0
        self.max_ep_reward, self.min_ep_reward, self.avg_ep_reward = 0, 0, 0
        self.num_game = 0

    def load_mem(self):
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        # saving and loading nn
        checkpoint = tf.train.get_checkpoint_state("saved_nn")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully Import!", checkpoint.model_checkpoint_path)
        else:
            print("Import Failed!")

    def save_mem(self):
        if self.model == MODEL_TRAINING:
            if self.t % 10000 == 0:
                self.saver.save(self.sess, 'saved_nn/' + GAME + '-dqn', global_step = self.t)

    def update_mm(self, r_current, s_next, terminal):
        if self.model == MODEL_TRAINING:
            self.Memory.append((self.s_current, self.actions, r_current, s_next, terminal))
            if len(self.Memory) > ROBOT_MEM:
                self.Memory.popleft()

    def train_nn(self, readout, y, a, s, train_step, s_next, cost):
        bval = True
        rcost = 0
        if bval == True and self.model == MODEL_TRAINING:
            if self.t > VIEW:
                minibatch = random.sample(self.Memory, BATCH)

                batch_current_state = [d[0] for d in minibatch]
                batch_action = [d[1] for d in minibatch]
                batch_reward = [d[2] for d in minibatch]
                batch_next_state = [d[3] for d in minibatch]

                batch_real_value = []
                readout_batch_next_state = readout.eval(session=self.sess, feed_dict={s: batch_next_state})
                for i in range(0, len(minibatch)):
                    terminal = minibatch[i][4]

                    if terminal:
                        batch_real_value.append(batch_reward[i])
                    else:
                        batch_real_value.append(batch_reward[i] + GAMMA * np.max(readout_batch_next_state[i]))

                with tf.device('/gpu:0'):
                    _, rcost = self.sess.run([train_step, cost], feed_dict={
                        y: batch_real_value,
                        a: batch_action,
                        s: batch_current_state
                    })

        self.s_current = s_next

        return rcost

    def choose_action_trainning(self, readout, s):
        readout_v = readout.eval(session=self.sess, feed_dict={s: [self.s_current]})[0]
        self.actions = np.zeros([ACTIONS])
        if self.model == MODEL_TRAINING:
            self.action_index = 0
            if self.t % ACTION_FRAME == 0:
                if random.random() <= self.random_threshold:
                    self.action_index = random.randrange(ACTIONS)
                    self.actions[random.randrange(ACTIONS)] = 1
                else:
                    self.action_index = np.argmax(readout_v)
                    self.actions[self.action_index] = 1
            else:
                self.actions[0] = 1 # do nothing
                print("nothing")
        elif self.model == MODEL_PLAY:
            # real play
            self.action_index = np.argmax(readout_v)
            self.actions[self.action_index] = 1

        return readout_v

    def initial_observe(self):
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        x_t, r_0, terminal = self.game.StepAgent(do_nothing)
        x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
        self.s_current = np.stack((x_t, x_t, x_t, x_t), axis=2)

    def create_next_Observe(self):
        x_t1_colored, r_current, terminal = self.game.StepAgent(self.actions)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_next_state = np.append(x_t1, self.s_current[:, :, :3], axis=2)
        return s_next_state, r_current, terminal

    def print_info(self, action_index, r_current, readout_v):
        state_info = ""
        if self.t <= VIEW:
            state_info = "view"
        elif self.t > VIEW and self.t <= VIEW + ADVENTURE:
            state_info = "adventure"
        else:
            state_info = "train"

        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_p:
                    # plot_cost()
                    print("TIMESTEP", self.t, "/ STATE", state_info, \
                          "/ RANDOM_THRESHOLD", self.random_threshold, "/ ACTION", action_index, "/ REWARD", r_current, \
                          "/ Q_MAX %e" % np.max(readout_v))

                if event.key == pygame.K_q:
                    self.plot_cost()
                    break;

    def plot_cost(self):
        #plt.ion()
        plt.plot(np.arange(len(cost_his)), cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show(block = True)

    def weight_variable(self, shape):
        #initialize the weights
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        #initialize the biases
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        #(W−F+2P)/S+1
        return tf.nn.conv2d(x, W, strides=[1, stride, stride,1], padding= "SAME")

    def max_pool_2x2(self, x):
        # 2*2 filters with stride 2
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    def nn_layer(self, input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = self.weight_variable([input_dim, output_dim])
                #self.variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = self.bias_variable([output_dim])
                #self.variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                #tf.summary.histogram('pre_activations', preactivate)

            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    def out_layer(self, input_tensor, input_dim, output_dim, layer_name):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = self.weight_variable([input_dim, output_dim])
                #self.variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = self.bias_variable([output_dim])
                #self.variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                #tf.summary.histogram('pre_activations', preactivate)

            activations = preactivate
            return activations

    def cnn_layer(self, input_tesor, filter, strides, input_dim, output_dim, layer_name, act=tf.nn.relu):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = self.weight_variable([filter, filter, input_dim, output_dim])
            with tf.name_scope('biases'):
                biases = self.bias_variable([output_dim])
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.nn.relu(self.conv2d(input_tesor, weights, strides) + biases)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    def createNetwork(self):
        # input layer

        with tf.name_scope('xts_inputs'):
            s = tf.placeholder("float", [None, 80, 80, 4], name='input')

        # hidden layers
        # filter stride input output
        # 20 * 20 * 32
        conv1 = self.cnn_layer(s, 8, 4, 4, 32, "xts_conv1")
        # 10 * 10 * 32
        h_pool1 = self.max_pool_2x2(conv1)
        # 5 * 5 * 64
        conv2 = self.cnn_layer(h_pool1, 4, 2, 32, 64, "xts_conv2")
        # 5 * 5 * 64
        conv3 = self.cnn_layer(conv2, 3, 1, 64, 64, "xts_conv3")
        # 1600
        h_conv3_flat = tf.reshape(conv3, [-1, 1600])
        h_fc1 = self.nn_layer(h_conv3_flat, 1600, 512, "xts_full_connection")

        # readout layer
        readout = self.out_layer(h_fc1, 512, 3, "xts_Out_put")

        self.h_fc1 = h_fc1

        return s, readout, h_fc1

    def trainNetwork(self, state, readout, h_fc1, sess):
        # define the cost function
        with tf.name_scope('cross_entropy'):
            action = tf.placeholder("float", [None, ACTIONS])
            real_lable = tf.placeholder("float", [None])
            readout_action = tf.reduce_sum(tf.multiply(readout, action), reduction_indices=1)
            with tf.name_scope('total'):
                cross_entropy = tf.reduce_mean(tf.square(real_lable - readout_action))

        tf.summary.scalar('cross_entropy', cross_entropy)

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)

        #################################
        with tf.variable_scope('summary'):
            scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
                                   'episode.max reward', 'episode.min reward', 'episode.avg reward',
                                   'episode.num of game', 'training.learning_rate']

            self.summary_placeholders = {}
            self.summary_ops = {}
            self.env_name = "Game Name"
            self.env_type = "plane"

            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag] = tf.summary.scalar("%s-%s/%s" % (self.env_name, self.env_type, tag),
                                                          self.summary_placeholders[tag])
        #################################
        #self.merged = tf.summary.merge_all()
        if self.model == MODEL_TRAINING:
            self.writer = tf.summary.FileWriter("logs/", self.sess.graph)

        self.load_mem()

        self.initial_observe()

        while True:
            readout_v = self.choose_action_trainning(readout, state)

            if self.random_threshold > FINAL_RANDOM_THRESHOLD and self.t > VIEW:
                self.random_threshold -= (self.initial_random_threshold - FINAL_RANDOM_THRESHOLD) / ADVENTURE

            s_next, r_current, terminal = self.create_next_Observe()

            self.update_mm(r_current, s_next, terminal)

            avg_loss = self.train_nn(readout, real_lable, action, state, train_step, s_next, cross_entropy)

            ###################################################
            self.inject_summary({
                'average.reward': self.avg_reward,
                'average.loss': avg_loss,
                # 'average.q': avg_q,
                'episode.max reward': self.max_ep_reward,
                'episode.min reward': self.min_ep_reward,
                'episode.avg reward': self.avg_ep_reward,
                'episode.num of game': self.num_game,
          }, self.t)
            ###################################################
            self.t += 1
            ###################################################
            self.total_reward += r_current
            self.avg_reward = self.total_reward / self.t
            if terminal:
                self.num_game += 1
                self.ep_rewards.append(self.ep_reward)
                self.ep_reward = 0.
            else:
                self.ep_reward += r_current

            try:
                self.max_ep_reward = np.max(self.ep_rewards)
                self.min_ep_reward = np.min(self.ep_rewards)
                self.avg_ep_reward = np.mean(self.ep_rewards)
            except:
                self.max_ep_reward, self.min_ep_reward, self.avg_ep_reward = 0, 0, 0
            ###################################################
            self.save_mem()
            # print info
            self.print_info(self.action_index, r_current, readout_v)

    def inject_summary(self, tag_dict, step):
        if self.model == MODEL_TRAINING:
            summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
                self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
            })
            for summary_str in summary_str_lists:
                self.writer.add_summary(summary_str, self.t)

    def run(self):
        s, readout, h_fc1 = self.createNetwork()
        self.trainNetwork(s, readout, h_fc1, self.sess)

def PlayeGame():
    nnk = Neuro_Network()
    nnk.run()

if __name__ == "__main__":
    PlayeGame()










































