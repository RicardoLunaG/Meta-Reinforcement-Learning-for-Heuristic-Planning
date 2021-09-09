
import numpy as np
import tensorflow as tf
import copy

from RL2_Model import PPOTrain
from Blocks_Env import Blocks_Env

from scipy.stats import entropy
from utils import PriorityQueue


class MetaTester:

    
    def __init__(self, task, env, model, state_space, action_space, model_path,sess):
        """
        Args:
            task: String, Desired name for the cluster of tasks
            env: Class, Enviroment to do the test on
            model: Class, Model to train on
            source_tasks: List, List of task to train in
            state_space: Integer, State space of the learning domain
            action_space: Integer, Action space of the learning domain
            model_path: String, Route to load the model
        """
        self.action_space = action_space
        self.state_space = state_space
        self.model_path = model_path
        self.model = model
        self.rnn_state = None
        self.task = task
        self.env = env
  
        self.action = 0
        self.time_steps = 0
        self.sess = sess


        self.PPO = self.model(self.state_space, self.action_space)
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=7)

        

        saver.restore(self.sess, model_path)


    def reset(self):
        self.rnn_state = None
        self.time_steps = 0
              
    def Test(self, state, action, reward, time_step = 0, rnn_state = None, rnn_state2 = None, rnn_state_state = 1, rnn_state_state2 = 1):
        """
        Args:
            
            state: Float, Current state
            action: Int, Current action
            reward: Float, Current reward
            
        """

    
        reward = 0
        
        if rnn_state_state == None:
            rnn_state = self.PPO.Policy.state_init
        
        if rnn_state_state2 == None:
            rnn_state2 = self.PPO.Policy.state_init2
        
        state = np.stack([state]).astype(dtype=np.float32)
        
        action, value, rnn_state, rnn_state2 = self.PPO.Policy.act(state, rnn_state, rnn_state2, reward, action, time_step, 1, False)
            
        return action, value, rnn_state, rnn_state2

    def get_action_probs(self, state, action, reward, time_step = 0, rnn_state = None, rnn_state2 = None):

        state = np.stack([state]).astype(dtype=np.float32)
        action_probs = self.PPO.Policy.get_action_probs(state, rnn_state, rnn_state2, reward, action, time_step)

        return action_probs
        



class ForwardSearch():

    def __init__(self,state_space,action_space,model_path):

        self.model_path = model_path
        self.state_space = state_space
        self.action_space = action_space
        """
        Args:
            state_space: Integer, State space of the learning domain
            action_space: Integer, Action space of the learning domain
            model_path: String, Route to load the model
        """

    test = 1

    
    def reconstruct_path(self,came_from, start, goal):
        current = goal
        path = []
        states = []
        rewards = []
        plan = []
        entropy_values = []
        while current.get_id() != start.get_id():
            path.append(current.action_took)
            states.append(current.getStateRepresentation())
            rewards.append(current.reward)
            entropy_values.append(current.entropy_value)
            current = came_from[current.get_id()]
        
        start.reset()

        path.reverse()
        for a in path:
            verbose = start.action_verbose(a)

            plan.append(verbose)
        
        
        states.append(start.getStateRepresentation())
        states.reverse()
        rewards.append(start.reward)
        rewards.reverse()
        return states,path,rewards

    def a_star_search(self,start,load_step):

        
        save_dir = '/'.join(self.model_path.split('/')[:-1])
        ckpt = tf.train.get_checkpoint_state(save_dir)
        model_path = ckpt.model_checkpoint_path
    
        state = start.getStateRepresentation()


        
        tf.reset_default_graph()

        with tf.Session() as sess:

            meta_tester = MetaTester("P1",env,18,PPOTrain,self.state_space,self.action_space,model_path,sess)
            for epi in range(1):
                visited_states = 0
                start_ = copy.deepcopy(start)
                frontier = PriorityQueue()
                frontier.put(start_, 0)
                came_from = {}
                cost_so_far = {}
                value_action = {}
                ff_action = {}
                grid_representations = {}
                came_from[start.get_id()] = None
                cost_so_far[start.get_id()] = 0
                if epi == 0:
                    start_.prefered_action, start_.value, start_.rnn_state, start_.rnn_state2 = meta_tester.Test(state, start_.action_took,start_.reward, time_step= start_.time_step, rnn_state_state=None,rnn_state_state2=None)
                



                found_valuable_action = False
                curret_ns = None

                current = None
                entropies = []
                while not frontier.empty():
                    
                    if found_valuable_action or curret_ns == None:
                        current = frontier.get()
                    else:
                        current = copy.deepcopy(curret_ns)
                        
                    curret_ns = None
                
                    found_valuable_action = False
                    
        
                    visited_states += 1
                    grid_representations[current.get_id()] = current.getGridRepresentation()

                    neighbors = current.get_neighbors()
                
                    if current.done:
                        break
                    pv_action = 1000

                    v_action = 0
                    f_action = 0
                    t = 0

                    for next in neighbors:
                        

                        t+=1

                        fp = 0
                        state = next.getStateRepresentation()
                        reward = next.reward
                        next.reward = reward + fp

                        action = next.action_took

                        action_would_take, value, rnn_state, rnn_state2 = meta_tester.Test(state, action, (reward+fp),  time_step= next.time_step, rnn_state=current.rnn_state, rnn_state2 = current.rnn_state2)
                        action_probs = meta_tester.get_action_probs(state, action, (reward+fp),  time_step= next.time_step, rnn_state=current.rnn_state, rnn_state2 = current.rnn_state2)
                    
                        entropy_value = entropy(action_probs[0])
                        entropies.append(entropy_value)
                    

                        value = np.asscalar(value)

                        if abs(value) < pv_action:
                            v_action = t
                            pv_action = abs(value)

                        next.prefered_action = action_would_take
                        next.value = value
                        next.rnn_state = rnn_state
                        next.rnn_state2 = rnn_state2
                        next.previous_state = current.getStateRepresentation()
                        next.previous_value = current.value
                        next.prevous_time_step = current.time_step
                        next.entropy_value = entropy_value


                        new_cost = cost_so_far[current.get_id()] + 1
                        next_id = next.get_id()

                        
            
                        
                        if next_id not in cost_so_far or new_cost < cost_so_far[next_id]:
                            cost_so_far[next_id] = new_cost
                        
                            if action == current.prefered_action:
                                curret_ns = next
 

                            priority = new_cost + -(value) -reward
                    

                    
                            frontier.put(next, priority)

                            came_from[next_id] = current
                    ff_action[current.get_id()] = f_action
                    value_action[current.get_id()] = v_action


        
                start.rnn_state = start_.rnn_state
                start.rnn_state2 = start_.rnn_state2
                start.value = start_.value
                states_path,path, _ = self.reconstruct_path(came_from,start,current)
            
        

        frontier = None
        cost_so_far = None
        return came_from, cost_so_far,visited_states, current, grid_representations, states_path, path, np.average(entropies)

if __name__ == '__main__':

    saving_route = './results/' #Route to save results
    avg_visted_states = 0
    test = 1
    visited = []
    load = []
    model_path = "./rl2"+str(test)+"/"
    test_tasks = [] #Tasks to test the model on
    state_space = 0
    action_space = 0
    fs = ForwardSearch(state_space,action_space,model_path)
 
    for task in test_tasks:

        env = Blocks_Env(task) #Change to desired domain
        start = copy.deepcopy(env)

        came_from, cost_so_far, visited_states, last_state,grid_repre,states_path, path,entropie = fs.a_star_search(start)
        visited.append(visited_states)
        print("Task: {}, Visited States: {}, Plan Length: {}".format(task, visited_states, len(path)))
        
        came_from, cost_so_far, visited_states, last_state,grid_repre = None, None, None, None, None
 


    np.save(saving_route+"visited_states-"+str(test), visited)
