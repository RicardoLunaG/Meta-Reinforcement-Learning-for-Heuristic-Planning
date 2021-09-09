
from  pddlpy import DomainProblem
import numpy as np
import copy
import akro
import gym


class Blocks_Env(gym.Env):
    def __init__(self, task = 0,pddl_directory = ""):
        """
        Args:
            task (int): Task number to load
            pddl_directory: Dictory in which the pddl files are stored.
        """
        self.id_gen = 1

        self.id = 1
        
        self.domainfile = pddl_directory+'/domain.pddl'
        self.problemfile = pddl_directory+'/pg'+str(task)+'.pddl'
        self.domprob = DomainProblem(self.domainfile, self.problemfile)
        self.action_space_ = 13
        self.state_space_ = 34
        self.task = task
        self.block_letters = ["A","B","C","D"]

        self.state_dictionary = {}


        self.holding = ""
        self.hand_empty = False

        self.goals = {}

        self.reward = 0
        self.time_step = 0


        self.on_table = []
        self.on_table_init = []
        self.clear = []
        self.clear_init = []
        self.goals_init = {}
        self.on_blocks = {}
        self.on_blocks_init = {}


        self.action_took = 0
        self.episodes = 0
        self.max_epidoes = 500
        self.done_max_episodes = False
        self.is_done1 = False
        self.is_done2 = False

        for i in self.block_letters:
            for j in self.block_letters:
                if i == j:
                    continue
                dict_string = ""+i+j
                self.state_dictionary[dict_string] = 0
        
        for block in self.block_letters:
            self.on_blocks_init[block] = ""

        for block in self.block_letters:
            self.goals_init[block] = ""
        self.entropy_value = 0
        self.init_counter = 0
        self.initialize()
        self.reset_state()
        self.done = False

    @property
    def observation_space(self):
        """gym.spaces.Box: The observation space."""
        return akro.Box(low=0,high=50,shape=(34,),dtype=int)


    @property
    def action_space(self):
        """gym.spaces.Box: The action space."""
        return akro.Discrete(13)

    def step(self,action):

        self.episodes += 1
        self.time_step+=1
        reward = 0
        success = False
        success_push = False
        if action < 4:
            success = self.pick_up(self.block_letters[action])
        elif action < 8:
            success = self.stack(self.block_letters[action-4])
        elif action < 12:
            success = self.unstack(self.block_letters[action-8])
        elif action == 12:
            success = self.put_down()
        else:
           raise Exception("Action out of action space") 

        

    

        if success:
            reward = -1        

        else:

            reward = -1

        counter = 0
        for goal in self.goals_init:

            if self.goals_init[goal] == self.on_blocks[goal]:
                counter += 1
        if counter > self.init_counter:
            reward = 50
            self.init_counter +=1


        if self.on_blocks == self.goals_init:
            reward = 1000

            self.done = True

        elif self.episodes > self.max_epidoes:
            self.done = True


        
        state =  self.getStateRepresentation()

        infos = dict(succes=success)
        
        return state, reward, self.done, infos
    
    def action_verbose(self,action):

        self.episodes += 1
        self.time_step+=1
        
        composed_string = ""
        if action < 4:
            composed_string = "pick {}".format(self.block_letters[action])
            self.pick_up(self.block_letters[action])
        elif action < 8:
            composed_string = "stack {} {}".format(self.holding,self.block_letters[action-4])
            self.stack(self.block_letters[action-4])
        elif action < 12:
            on_block = self.on_blocks[self.block_letters[action-8]]
            composed_string = "unstack {} {}".format(self.block_letters[action-8], on_block)
            self.unstack(self.block_letters[action-8])
        elif action == 12:
            composed_string = "putdown {}".format(self.holding)
            self.put_down()
        else:
           raise Exception("Action out of action space") 

        
        return composed_string
        
    def getGridRepresentation(self):
        grid = np.zeros([len(self.block_letters),len(self.block_letters)+1],dtype=str)
        t = 0
        for tb in self.on_table:
            
            if tb not in self.on_blocks.values():
                grid[-1][t] = tb
                t+=1
        if self.holding != "":
            grid[0][0] = self.holding
        else:
            grid[0][0] = "Y"
        leng = 0
        max_key = ""
        for key in self.on_blocks:
            p_key = key
            p_leng = 0
            while p_key in self.on_blocks:
                p_key = self.on_blocks[p_key]
                p_leng += 1
            if p_leng > leng:
                leng = p_leng
                max_key=key
        y_len = len(self.block_letters)-(leng+1)
        if leng>0:
            for i in range(leng+1):
                grid[y_len][-1] = max_key
                if max_key in self.on_blocks:
                    max_key = self.on_blocks[max_key]
                y_len += 1        
        
        return grid
    def getStateRepresentation(self):
        
        state_dict = copy.deepcopy(self.state_dictionary)
        for i in self.on_blocks:
            if self.on_blocks[i] != "":
                dict_string = ""+str(i)+str(self.on_blocks[i])
                state_dict[dict_string] = 1
        
        goals_dict = copy.deepcopy(self.state_dictionary)
        
        for i in self.goals_init:
            if self.goals_init[i] != "":
                dict_string = ""+str(i)+str(self.goals_init[i])

                goals_dict[dict_string] = 1
        
            
        goals_dict_values = np.fromiter(goals_dict.values(),dtype=float)
        state_dict_values = np.fromiter(state_dict.values(), dtype=float)

        int_hand = int(self.hand_empty)
        holding_int = 0

        if self.holding != "":
            holding_int = self.block_letters.index(self.holding) + 1 

        on_table = np.zeros([len(self.block_letters)])

        for i in self.on_table:
            idx = self.block_letters.index(i)
            on_table[idx] = 1

        goals_zeros = np.zeros((len(self.goals_init)))
        counter = 0
        for goal in self.goals_init:
            if self.goals_init[goal] == self.on_blocks[goal]:
                goals_zeros[counter] = 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
            counter += 1

        

        state = np.concatenate([state_dict_values,on_table,[int_hand],[holding_int],goals_zeros,goals_dict_values])

        return state
   

    def reset(self):
        self.reset_state()
        self.init_counter = 0
        self.episodes = 0
        state =  self.getStateRepresentation()
        self.done = False

        self.is_done1 = False
        return state
    
    def reset_state(self):

        self.clear = copy.deepcopy(self.clear_init)
        self.on_table = copy.deepcopy(self.on_table_init)
        self.hand_empty = self.hand_empty_init
        self.on_blocks = copy.deepcopy(self.on_blocks_init)
        

       
    def get_neighbors(self):
        
        neighbors = []
        copy_env = copy.deepcopy(self)

        for action in range(self.action_space):
            
            copy_env = copy.deepcopy(self)
            copy_env.action_took = action
            _,reward,_,success = copy_env.step(action)
            copy_env.reward = reward
            if success:
                neighbors.append(copy_env)            

        return neighbors


    def pick_up(self, l_block):
        
        if not self.hand_empty:
            return False
        if l_block not in self.on_table:
            return False
        if l_block not in self.clear:
            return False
        self.on_table.remove(l_block)
        self.clear.remove(l_block)
        self.hand_empty = False
        self.holding = l_block
        
        return True
    
    def put_down(self):
        
        if self.holding == "":
            return False
        block = copy.copy(self.holding)

        self.holding = ""
        self.clear.append(block)
        self.on_table.append(block)
        self.hand_empty = True
        
        return True
    
    def stack(self, l_block):
        
        if self.holding == "":
            return False
        
        if l_block not in self.clear:
            return False
        block = copy.copy(self.holding)
        self.holding = ""
        
        self.clear.append(block)
        self.clear.remove(l_block)
        self.on_blocks[block] = l_block
        self.hand_empty = True
        
        return True
    
    def unstack(self, l_block):
        
        if not self.hand_empty:
            return False
        if l_block not in self.on_blocks:
            return False
        if l_block not in self.clear:
            return False
       

        self.holding = l_block
        block = copy.copy(self.on_blocks[l_block])
        self.clear.append(block)
        self.clear.remove(l_block)
        self.hand_empty = False
        self.on_blocks[l_block] = ""
        return True
    
    
    def initialize(self):

        for s in self.domprob.initialstate():
            s_ground = s.ground([])

            if ("CLEAR") in s_ground:

                self.clear_init.append(s_ground[1])
            elif ("ONTABLE") in s_ground:

                self.on_table_init.append(s_ground[1])
            elif ("HANDEMPTY") in s_ground:

                self.hand_empty_init = True
            elif ("ON") in s_ground:
                self.on_blocks_init[s_ground[1]] = s_ground[2] 

        for s in self.domprob.goals():
            
            s_ground = s.ground([])
            
            self.goals_init[s_ground[1]] = s_ground[2]
        

    
    def __lt__(self, other):
        return self.action_space < other.action_space


    def get_id(self):
        id_o = self.getStateRepresentation()
        return id_o.tostring()
    
    def get_id_param(self, state):
        return state.tostring()
        

