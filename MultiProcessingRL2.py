import numpy as np
import tensorflow as tf
import copy
import numpy as np
import time
from multiprocessing import Process, Pipe
import threading

from RL2_Model import PPOTrain
from Blocks_Env import Blocks_Env

import time


class MasterProcess():
    def __init__(self, source_tasks,save_path, n_process = 8,n_per_process = 1,test=0):
        self.processes = {}
        self.n_process = n_process
        self.n_per_process = n_per_process
        self.source_tasks = source_tasks
        self.save_path = save_path
        self.test = test

    def train_agents(self):
        pipes = {}


        for i in range(0, self.n_process):
            parent_conn, child_conn = Pipe()
            pipes[i] = parent_conn

            p = AgentProcess(conn=child_conn, id=i, n_games=self.n_per_process, source_tasks = self.source_tasks[i],save_path=self.save_path)
            p.start()
            self.processes[i] = p

        scores = {}
        batchs = {}
        t0 = time.time()
        def listenToAgent(id, scores):
            while True:
                msg = pipes[id].recv()
                if msg == "saved":

                    for j in pipes:
                        if(j != 0):
                            pipes[j].send("load")
                else:
                    score = float(msg[0])
                    scores[id] = score
                    batchs[id] = msg[1]


        threads_listen = []
        print("Threads to start")
        for id in pipes:
            t = threading.Thread(target=listenToAgent, args=(id,scores))
            t.start()
            threads_listen.append(t)
        print("Threads started")

        window = 75 
        iter = 1
        mean_scores = []
        file = open("log_scores"+str(self.test), "w")
        while True:
            if(len(scores) == self.n_process-1):
                if max(scores) > 100:
                    id_best = max(scores, key=scores.get)
                    file.write("Reward Done : "+str(id_best)+"\n")
                mean_scores.append(np.mean(list(scores.values())))
                print("Test "+str(self.test)+" End of iteration "+str(iter)+". Mean score sor far : "+str(np.mean(mean_scores)))
                iter += 1
                file.write("Iterarion: "+ str(iter)+" Score: "+str(np.mean(mean_scores))+"\n")

                file.flush()

                pipes[0].send(("train_with_batchs", list(batchs.values())))
                t0 = time.time()
                scores.clear()
                batchs.clear()

            if(len(mean_scores) >= window):
                mean_scores = mean_scores[1:]

class AgentProcess(Process):
    def __init__(self, conn, id, n_games,source_tasks,save_path):
        super(AgentProcess,self).__init__()
        self.conn = conn
        self.n_games = n_games
        self.id = id
        self.msg_queue = []
        np.random.seed(self.id*100)
        self.save_path = save_path #Route to save the model in.

        self.source_task = source_tasks 



    def run(self):
        state_space = 34
        action_space = 13
        self.agent = PPOTrain(state_space,action_space,self.save_path)
        env = self.source_task

        def treatQueue():
            msg = self.conn.recv()
            if msg == "load":
                self.agent.load_model()


            if msg[0] == "train_with_batchs":

                t0 = time.time()
                self.agent.train_with_batchs(msg[1])
                self.agent.save_model()

                self.conn.send("saved")

        while True:
            if(self.id != 0):
                

                scores = []
                overall_data = 0
                states = []
                actions = []
                values = []
                rewards = []
                v_next = []

                index_source = 0
                time_steps = []

                for i in range(self.n_games):

                    reward = 0
                    action = 0 
                    visited_states = 0
                    t = 0

                    
                    rnn_state = self.agent.Policy.state_init
                    rnn_state2 = self.agent.Policy.state_init2

                    index_source += 1

                    
                    env_ = copy.deepcopy(env)

                    state = env_.reset()
         
                    ts = 0

                    scores_ep = 0
                    visited_states = 0
                    done = False
                    reward = 0
                    action = 0
                    time_step = 0
                    while not done:

                        visited_states += 1
                        time_step += 1
                       
                        ts += 1
                       
                        fp = 0



                        state = np.stack([state]).astype(dtype=np.float32)

                        action, value, rnn_state, rnn_state2 = self.agent.Policy.act(state, rnn_state, rnn_state2, reward, action,time_step, 1, True)

                        action = np.asscalar(action)

                        next_state, reward, done, _ = env_.step(action)
                        value = np.asscalar(value)

                        states.append(state)
          
                        fp = 0

                        reward = reward + fp
                        rewards.append(reward)

                        actions.append(action)

                        values.append(value)

                        time_steps.append(time_step)

                        if done:
                            v_next = values[1:] + [0]
                       
                        state = next_state
                        scores_ep += reward
                    
                        
          
                    scores.append(scores_ep)
                    overall_data += t
                    self.agent.assign_policy_parameters()

                batch = (states, actions, rewards,values,v_next,time_steps)
                self.conn.send((np.mean(scores),batch))
            treatQueue()


if __name__ == '__main__':


    run_number = 1
    num_game_per_process = 10
    sourceTask = []
    sourceTask_Graph = []
    
    pddl_path = "./pddl_files" #Pddl files directory
    save_path = "./rl2/" #Route to save the model in.
    num_of_tasks = 20

    for i in range(num_of_tasks):

        sourceTask.append(Blocks_Env(i),pddl_path) #Change to the desired domain


    master = MasterProcess(source_tasks = sourceTask,save_path=save_path,num_games_per_process=num_game_per_process,n_process=len(sourceTask),test=run_number)
    master.train_agents()

            