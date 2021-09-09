import numpy as np
import tensorflow as tf
import copy



class Model:
    def __init__(self,scope:str,state_space,action_space, hidden_layers = 64):

        """
        Args:
            state_space: Integer, State space of the learning domain
            action_space: Integer, Action space of the learning domain
            hidden_layers: Integer, Desired number of nodes in hidden layers

        """
        self.sess = None
        with tf.variable_scope(scope):

            self.state = tf.placeholder(shape=[None,state_space],dtype=tf.float32, name="state")
            self.prev_rewards = tf.placeholder(shape=[None,1],dtype=tf.float32, name="prev_rewards")
            self.prev_actions = tf.placeholder(shape=[None],dtype=tf.int32, name="prev_actions")
            self.timestep = tf.placeholder(shape=[None,1],dtype=tf.float32, name="time_step")
            self.prev_actions_onehot = tf.one_hot(self.prev_actions,action_space,dtype=tf.float32)
            
            hidden = tf.concat([self.state, self.prev_rewards,self.prev_actions_onehot,self.timestep],1)
            

            lstm_cell = tf.contrib.rnn.GRUBlockCellV2(hidden_layers, name = 'RNNCellC1_') 
            c_init = np.zeros((1, lstm_cell.state_size), np.float32)
         
            self.state_init = c_init
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size], name="c_in1")
            self.state_in = c_in
            
            rnn_in = tf.expand_dims(hidden,[0])
            step_size = tf.shape(self.prev_rewards)[:1]
            state_in = (c_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state = state_in, sequence_length = step_size, time_major = False)

            lstm_c = lstm_state
            self.state_out = lstm_c[:1, :]
            rnn_out = tf.reshape(lstm_outputs, [-1, hidden_layers])

            lstm_cell2 = tf.contrib.rnn.GRUBlockCellV2(hidden_layers, name = 'RNNCellC2_') 
            c_init2 = np.zeros((1, lstm_cell2.state_size), np.float32)
           
            self.state_init2 = c_init2
            c_in2 = tf.placeholder(tf.float32, [1, lstm_cell2.state_size], name="c_in2")
            self.state_in2 = c_in2
            
            rnn_in2 = tf.expand_dims(hidden,[0])
            step_size = tf.shape(self.prev_rewards)[:1]
            state_in2 = (c_in2)
            lstm_outputs2, lstm_state2 = tf.nn.dynamic_rnn(lstm_cell2, rnn_in2, initial_state = state_in2, sequence_length = step_size, time_major = False)

            lstm_c2 = lstm_state2
            self.state_out2 = lstm_c2[:1, :]
            rnn_out2 = tf.reshape(lstm_outputs2, [-1, hidden_layers])

            with tf.variable_scope('Policy_Estimator'):
                
                l2 = tf.layers.dense(inputs=rnn_out,units = hidden_layers, activation = tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
                self.action_probs = tf.layers.dense(inputs=l2, units=action_space, activation=tf.nn.softmax)
                
            with tf.variable_scope('Value_Estimator'):
     
                l2 = tf.layers.dense(inputs=rnn_out2, units = (hidden_layers), activation=tf.nn.tanh,kernel_initializer=tf.variance_scaling_initializer(scale=2))
                self.value = tf.layers.dense(inputs = l2, units = 1, activation = None)
            
            self.act_stochastic = tf.multinomial(tf.log(self.action_probs), num_samples = 1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.action_probs, axis = 1)

            self.scope = tf.get_variable_scope().name

    def act(self, states, rnn_state, rnn_state2,reward, action, t, column = 1, stochastic = True):
        sess = self.sess
        
        if column == 1: 
            if stochastic:
                return sess.run([self.act_stochastic,self.value,self.state_out,self.state_out2], feed_dict={ self.state:states,
                    self.state_in: rnn_state,
                    self.state_in2: rnn_state2,
                    self.prev_rewards:[[reward]],
                    self.timestep:[[t]],
                    self.prev_actions:[action]})
            else:
                return sess.run([self.state_out, self.act_deterministic,self.value], feed_dict={self.state_in[0]: rnn_state[0],self.state_in[1]: rnn_state[1], self.state:states})
    
    def get_action_probs(self,states, rnn_state, rnn_state2, reward, action, t, column = 1):
        sess = self.sess
        if column == 1:
            return sess.run(self.action_probs, feed_dict={ self.state:states,
                    self.state_in: rnn_state,
                    self.state_in2: rnn_state2,
                    self.prev_rewards:[[reward]],
                    self.timestep:[[t]],
                    self.prev_actions:[action]})
    
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    
    
class PPOTrain:
    def __init__(self,state_space, action_space, save_path = "", gamma = 0.99, clip_value = 0.2, c_1 = 0.5, c_2=0.05,learning_rate = 0.0001): 

        """
        Args:
            state_space: Integer, State space of the learning domain
            action_space: Integer, Action space of the learning domain
            hidden_layers: Integer, Desired number of nodes in hidden layers
            save_path: String, Route to save the model
            gamma: Float, Discount rate
            clip_value: Float, Clip value for the PPO objective
            c_1: Float, Weight of value loss
            c_2: Float, Weight of entropy loss
            learning_rate: Float, Learning rate


        """

        self.Policy = Model('Policy', state_space, action_space)
        self.Old_Policy = Model('Old_Policy', state_space, action_space)
        self.state_space = state_space
        self.save_path = save_path
        
        self.gamma = gamma
        self.squared_loss = 0



        trainable = self.Policy.get_trainable_variables()

        pi_trainable = trainable
       

        old_trainable = self.Old_Policy.get_trainable_variables()

        old_pi_trainable = old_trainable
        
        
        with tf.variable_scope('Assign_Operator'):
            self.assign_ops = []
            for v_old, v in zip(old_pi_trainable, pi_trainable):
                self.assign_ops.append(tf.assign(v_old,v))
        
        with tf.variable_scope('Inputs_Train'):
            self.actions = tf.placeholder(tf.int32, [None], "Actions")
            self.rewards = tf.placeholder(tf.float32, [None], "Rewards")
            self.v_next = tf.placeholder(tf.float32, [None], "Value_Next")
            self.gaes = tf.placeholder(tf.float32, [None], "GAES")

        
        action_probs = self.Policy.action_probs
        action_probs_old = self.Old_Policy.action_probs
        


        action_probs = action_probs * tf.one_hot(indices = self.actions, depth = action_probs.shape[1])
        action_probs = tf.reduce_sum(action_probs, axis = 1)


        action_probs_old = action_probs_old * tf.one_hot(indices = self.actions, depth = action_probs_old.shape[1])
        action_probs_old = tf.reduce_sum(action_probs_old, axis = 1)

        
        ratios = tf.exp(tf.log(action_probs)-tf.log(action_probs_old))
        clipped_ratios = tf.clip_by_value(ratios, clip_value_min = 1 - clip_value, clip_value_max = 1 + clip_value)
        loss_clip = tf.minimum(tf.multiply(self.gaes,ratios),tf.multiply(self.gaes,clipped_ratios))
        loss_clip = tf.reduce_mean(loss_clip)
        tf.summary.scalar('loss_clip', loss_clip)

        
        v_preds = self.Policy.value
        
        loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_next, v_preds)
        self.squared_loss = tf.squared_difference(self.rewards + self.gamma * self.v_next, v_preds)
        loss_vf = tf.reduce_mean(loss_vf)
        
        tf.summary.scalar('loss_vf',loss_vf)

        
        a_probs = self.Policy.action_probs
        
        entropy = - tf.reduce_sum(a_probs * tf.log(tf.clip_by_value(a_probs, 1e-10, 1.0)), axis = 1)
        entropy = tf.reduce_mean(entropy, axis = 0)
        tf.summary.scalar("Entropy",entropy)
        
        
        loss = loss_clip - c_1 * loss_vf + c_2 * entropy
        self.loss = -loss
        tf.summary.scalar('Loss',self.loss)
        
        self.merged = tf.summary.merge_all()
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon = 1e-3) 
        self.train_op = optimizer.minimize(self.loss, var_list = pi_trainable)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        
        config = tf.ConfigProto()

        self.sess = tf.Session(config=config)
        self.initializer = tf.global_variables_initializer()
        self.sess.run(self.initializer)
        
        self.saver = tf.train.Saver()
        self.Policy.sess = self.sess
        self.Old_Policy.sess = self.sess
        
        self.load_model()
    
    def train(self, states, actions, rewards, v_next, gaes, time_steps):
        sess = self.sess
        prev_rewards = [0] + rewards[:-1].tolist()
        prev_actions = [0] + actions[:-1].tolist()

        rnn_state = self.Policy.state_init
        rnn_state2 = self.Policy.state_init2
        rnn_state_old = self.Old_Policy.state_init
        rnn_state_old2 = self.Old_Policy.state_init2
        
        return sess.run([self.merged, self.squared_loss, self.train_op, self.Policy.state_out, self.Policy.state_out2], feed_dict={
            self.Policy.state:states,
            self.Old_Policy.state:states,
            self.actions: actions,
            self.rewards: rewards,
            self.v_next: v_next,
            self.gaes: gaes,
            self.Policy.timestep: time_steps,
            self.Old_Policy.timestep: time_steps,

            self.Policy.state_in:rnn_state,
            self.Policy.state_in2:rnn_state2,

            self.Policy.prev_rewards:np.vstack(prev_rewards),
            self.Policy.prev_actions:prev_actions,
            
 

            self.Old_Policy.prev_rewards:np.vstack(prev_rewards),
            self.Old_Policy.prev_actions:prev_actions,

            self.Old_Policy.state_in:rnn_state_old,
            self.Old_Policy.state_in2:rnn_state_old2,
        })
    def get_summary(self, states, actions, rewards, v_next, gaes):
        sess = self.sess
        return sess.run([self.merged], feed_dict = {
            self.Policy.state: states,
            self.Old_Policy.state: states,
            self.actions: actions,
            self.rewards: rewards,
            self.v_next: v_next,
            self.gaes: gaes
        })
    def assign_policy_parameters(self):
        sess = self.sess
        sess.run(self.assign_ops)

    def get_gaes(self,rewards, values, v_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_next, values)]
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes)-1)):
            gaes[t] = gaes[t] + self.gamma * gaes[t+1]
        return gaes
    
    def load_model(self):
        try:
            save_dir = '/'.join(self.save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            self.sess.run(self.initializer)
            self.saver.restore(self.sess, load_path)
        except Exception as e:
            print(e)
            print("No saved model to load, starting a new model from scratch.")

    
    def train_with_batchs(self, batch):

        
        self.batch_states = []
        self.batch_values = []
        self.batch_actions = []
        self.batch_v_next = []
        self.batch_rewards = []
        self.time_steps = []

        for x in batch:
            self.batch_states = x[0]
            self.batch_actions = x[1]
            self.batch_rewards = x[2]
            self.batch_values = x[3]
            self.batch_v_next = x[4]
            self.time_steps = x[5]
        
            gaes = self.get_gaes( self.batch_rewards, self.batch_values, self.batch_v_next)

            self.batch_states = np.array(self.batch_states)
            self.batch_states = np.reshape(self.batch_states, newshape = [-1,self.state_space])


            self.batch_actions = np.array(self.batch_actions).astype(dtype=np.int32)
            self.batch_rewards = np.array(self.batch_rewards).astype(dtype=np.float32)
            self.batch_v_next = np.array(self.batch_v_next).astype(dtype=np.float32)

            self.time_steps = np.array(self.time_steps).astype(dtype=np.int32)
            self.time_steps = np.vstack(self.time_steps)
            gaes = np.array(gaes).astype(dtype=np.float32)

            gaes = (gaes - gaes.mean()) / max(gaes.std(),0.0001)

      
            self.train(self.batch_states,self.batch_actions,self.batch_rewards,self.batch_v_next, gaes,self.time_steps)



    def save_model(self):
        self.saver.save(self.sess, self.save_path, global_step=self.global_step)