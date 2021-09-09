
from  pddlpy import DomainProblem
import akro
import gym
import numpy as np
import copy


class Ferry_Env(gym.Env):
    def __init__(self, task = 0, pddl_directory = ""):

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
        self.num_locations = 4
        self.num_cars = 4

        self.task = task
        self.agent_pos = ""
        self.obj_positon = []
        self.adjacent_points = []
        
        self.locations = []
        self.car_locations = []
        self.reward = 0
        self.time_step = 0
        self.agent_pos_init = ""
        self.adjacent_points_init = []
        self.car_locations_init = {}
        self.prefered_action = 0
        self.goals = {}      
        self.on_car = ""
        self.action_took = 0
        self.episodes = 0
        self.max_epidoes = 500
        self.done_max_episodes = False
        self.empty_ferry = False
        self.entropy_value = 0

        self.initialize()
        self.reset_state()
        self.init_counter = 0
        self.done = False
    
    @property
    def observation_space(self):
        """gym.spaces.Box: The observation space."""
        return akro.Box(low=0,high=50,shape=(14,),dtype=int)


    @property
    def action_space(self):
        """gym.spaces.Box: The action space."""
        return akro.Discrete(9)

    def step(self,action):
       
        self.episodes += 1
        self.time_step+=1
        
        reward = 0
        success = False
        if action == 0:

            success = self.sail("l0")
        elif action == 1:
            success = self.sail("l1")
        elif action == 2:
            success = self.sail("l2")
        elif action == 3:
            success = self.sail("l3")

        elif action == 4:
            success = self.board("c0")
        elif action == 5:
            success = self.board("c1")
        elif action == 6:
            success = self.board("c2")
        elif action == 7:
            success = self.board("c3")
        elif action == 8:
            success = self.debark()
        else:
           raise Exception("Action out of action space") 


        

        if success:
            reward = -1
        else:
            reward = -1

        counter = 0
        for goal in self.goals:
            if self.goals[goal] == self.car_locations[goal]:
                counter += 1
        if counter > self.init_counter:
            reward = 50
            self.init_counter +=1
          
        if self.goals == self.car_locations:
         
            self.done = True
        elif self.episodes > self.max_epidoes:
            self.done = True
 


        state =  self.getStateRepresentation()
        info = success
        return state, reward, self.done, info
    
    def action_verbose(self,action):
       
        self.episodes += 1
        self.time_step+=1
        
        composed_string = ""
        if action == 0:
            composed_string = "sail {} {}".format(self.agent_pos,"l0")
            self.sail("l0")
        elif action == 1:
            composed_string = "sail {} {}".format(self.agent_pos,"l1")
            self.sail("l1")
        elif action == 2:
            composed_string = "sail {} {}".format(self.agent_pos,"l2")
            self.sail("l2")
        elif action == 3:
            composed_string = "sail {} {}".format(self.agent_pos,"l3")
            self.sail("l3")

        elif action == 4:
            composed_string = "board {} {}".format("c0",self.agent_pos)
            self.board("c0")
        elif action == 5:
            composed_string = "board {} {}".format("c1",self.agent_pos)
            self.board("c1")
        elif action == 6:
            composed_string = "board {} {}".format("c2",self.agent_pos)
            self.board("c2")
        elif action == 7:
            composed_string = "board {} {}".format("c3",self.agent_pos)
            self.board("c3")
        elif action == 8:
            composed_string = "debark {} {}".format(self.on_car,self.agent_pos)
            self.debark()
        else:
           raise Exception("Action out of action space") 


        return composed_string
    
    
    def getGridRepresentation(self):

        grid_state = np.zeros([2,self.num_locations])



        indx_ferry = int(self.agent_pos[1])
        

        for car in self.car_locations:
            if self.car_locations[car] != "Boat":
                grid_state[0][int(self.car_locations[car][1])] = int(car[1])+1

        car_on = 9 if self.on_car == "" else int(self.on_car[1]) + 1

        grid_state[1][indx_ferry] = car_on


        
        
        return grid_state
    def getStateRepresentation(self):
       
        one_hot_ferry_loc = np.zeros(self.num_locations)
        car_loc_vec = np.zeros(self.num_cars)

        goals_loc_vec = np.zeros(self.num_cars)

        indx_ferry = int(self.agent_pos[1])
        one_hot_ferry_loc[indx_ferry] = 1

        for car in self.car_locations:
            if self.car_locations[car] != "Boat":
                car_loc_vec[int(car[1])] = int(self.car_locations[car][1])+1
        
        for car in self.goals:
            if self.goals[car]:
                goals_loc_vec[int(car[1])] = int(self.goals[car][1])+1
        car_on = 0 if self.on_car == "" else int(self.on_car[1]) + 1
        
        goals_one_hot = np.zeros(len(self.goals))
        counter = 0
        
        for goal in self.goals:
            if self.goals[goal] == self.car_locations[goal]:
                goals_one_hot[counter] = 1
            counter += 1
        

        state = np.concatenate((one_hot_ferry_loc,car_loc_vec,[car_on],goals_loc_vec,goals_one_hot))

        return state
   

    def reset(self):
        self.reset_state()
        self.init_counter = 0
        self.episodes = 0
        state =  self.getStateRepresentation()
        self.done = False
        return state
  

    def reset_state(self):

        self.agent_pos = copy.deepcopy(self.agent_pos_init)
        self.init_counter = 0
        self.empty_ferry = self.empty_ferry_init
        self.car_locations = copy.deepcopy(self.car_locations_init)
        self.adjacent_points = self.adjacent_points_init
        
        

       
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




    def sail(self,l2):
        
        l1 = self.agent_pos
        if l2 == l1:
            return False

        is_adjacent = False

        for s in self.adjacent_points:

            if (l1,l2) == s:
                is_adjacent = True

            elif (l2,l1) == s:
                is_adjacent = True

        if not is_adjacent:
            return False
        self.agent_pos = l2
        
        return True
    
    def board(self, car):
        
        loc = self.agent_pos

        if not self.empty_ferry:
            return False

        if self.car_locations[car] != loc:
            return False
        self.on_car = car

        self.car_locations[car] = "Boat"
        self.empty_ferry = False

        return True
    
    def debark(self):
        
        loc = self.agent_pos
        if self.on_car == "":
            return False

        if self.empty_ferry:
            return False

        self.car_locations[self.on_car] = loc
            
        self.empty_ferry = True

        self.on_car = ""

        return True

        
        

    def initialize(self):
        for s in self.domprob.initialstate():
            s_ground = s.ground([])
            if ("location") in s_ground:
                self.locations.append(s_ground[1])
            elif ("not-eq") in s_ground:
                self.adjacent_points_init.append((s_ground[1],s_ground[2]))
            elif ("empty-ferry") in s_ground:
                self.empty_ferry_init = True
            elif ("at") in s_ground:
                self.car_locations_init[s_ground[1]] = s_ground[2]
            elif ("at-ferry") in s_ground:
                self.agent_pos_init = s_ground[1]


        for s in self.domprob.goals():
            s_ground = s.ground([])
            
            if len(s_ground) > 1:
                self.goals[s_ground[1]] = s_ground[2]

    
    def __lt__(self, other):
        return self.action_space < other.action_space

    def __eq__(self, other): 
        if not isinstance(other, Ferry_Env):

            return NotImplemented

        return (self.agent_pos == other.agent_pos and self.goals == other.goals)

    def get_id(self):
        id_o = self.getStateRepresentation()

        return np.array_str(id_o)
    
    def get_id_param(self, state):
        return state.tostring()


