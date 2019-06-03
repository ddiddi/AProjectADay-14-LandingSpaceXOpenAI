import tensorflow
import gym

class PPO(object):

    def __init__(self):
        self.session = tf.Session()
        self.tfs = tf.placeholder(tf.float32,[NONE,S_DIM], 'state')

        #critic
        with tf.variable_scope('critic')
            l1 = tf.layers.dense(self.tfs,100,tf.nn.relu)
            self.v = tf.layers(l1,1)

            self.tfdc_reward = tf.placeholder(tf.float32,[None,1],'discounter_reward')
            self.advantage = self.tfdc_reward - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)
