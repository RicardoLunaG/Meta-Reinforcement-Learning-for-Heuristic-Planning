from  pddlpy import DomainProblem
import numpy as np
import copy
import gym
import akro

class Sokoban_Env(gym.Env):
    def __init__(self, task = 0, pddl_directory = ""):

        """
        Args:
            task (int): Task number to load
            pddl_directory: Dictory in which the pddl files are stored.
        """

        self.id_gen = 1

        self.id = 1
        self.task = task
        self.domainfile = pddl_directory+'/domain.pddl'
        self.problemfile = pddl_directory+'/pg'+str(task)+'.pddl'
        self.domprob = DomainProblem(self.domainfile, self.problemfile)

        self.grid_size = 8
        self.task = task
        self.agent_pos = ""
        self.obj_positon = []
        self.adjacent_points = []
        self.adjacent_points_2 = []
        self.goals = []
        self.reward = 0
        self.time_step = 0
        self.agent_pos_init = ""
        self.obj_positon_init = []
        self.adjacent_points_init = []
        self.adjacent_points_2_init = []
        self.goals_init = []
        self.locations_init = np.zeros([self.grid_size,self.grid_size])
        self.action_took = 0
        self.episodes = 0
        self.max_epidoes = 500
        self.done_max_episodes = False
        self.is_done1 = False
        self.cheatAc=True
        self.is_done2 = False

        self.initialize()
        self.reset_state()
        self.done = False
        self.reward_range = [-1200.0,1000.0]

    @property
    def observation_space(self):
        """gym.spaces.Box: The observation space."""
        return akro.Box(low=0,high=11,shape=(70,),dtype=int)
 

    @property
    def action_space(self):
        """gym.spaces.Box: The action space."""
        return akro.Discrete(4)
    def step(self,action):

        self.episodes += 1
        self.time_step+=1
        reward = 0
        success = False
        success_push = False
        if action == 0:
            success = self.move(1,0)
            if not success:
                success = self.push(1,0)
        elif action == 1:
            success = self.move(-1,0)
            if not success:
                success = self.push(-1,0)
        elif action == 2:
            success = self.move(0,1)
            if not success:
                success = self.push(0,1)
        elif action == 3:
            success = self.move(0,-1)
            if not success:
                success = self.push(0,-1)
        else:
           raise Exception("Action out of action space") 

        

        

        if success_push:
            reward = -1        

        elif success:
            reward = -1

        else:
            reward = -1
        

        for obj in self.obj_positon:
            if not self.is_done1: 
                if self.goals[0] == obj:
                    self.is_done1=True
                    reward += 50
            if not self.is_done2:
                if self.goals[1] == obj:
                    self.is_done2=True 
                    reward += 50
        if self.goals[0] == self.obj_positon[0] and self.goals[1] == self.obj_positon[1]:
            self.done = True
            reward = 1000


        elif self.goals[1] == self.obj_positon[0] and self.goals[0] == self.obj_positon[1]:
            self.done = True
            reward = 1000

        

        state =  self.getStateRepresentation()
        infos = success
        return state, reward, self.done, infos
    
    def action_verbose(self,action):

        self.episodes += 1
        self.time_step+=1

        success = False

        composed_string = ""
        
        if action == 0:
            pos_1 = self.agent_pos
            success = self.move(1,0)
            pos_2 = self.agent_pos
            composed_string = "move {} {}".format(pos_1,pos_2)
            if not success:
                pos_1 = self.agent_pos
                self.push(1,0)
                pos_2 = self.agent_pos
                next_pos_2 = list(pos_1)
                next_pos_2[1] = str(int(next_pos_2[1])+2)
                next_pos_2[4] = str(int(next_pos_2[4]))
                next_pos_2 = "".join(next_pos_2)
                composed_string = "push {} {} {}".format(pos_1,pos_2,next_pos_2)
        elif action == 1:
            pos_1 = self.agent_pos
            success = self.move(-1,0)
            pos_2 = self.agent_pos
            composed_string = "move {} {}".format(pos_1,pos_2)
            if not success:
                pos_1 = self.agent_pos
                self.push(-1,0)
                next_pos_2 = list(pos_1)
                next_pos_2[1] = str(int(next_pos_2[1])-2)
                next_pos_2[4] = str(int(next_pos_2[4]))
                next_pos_2 = "".join(next_pos_2)
                composed_string = "push {} {} {}".format(pos_1,pos_2,next_pos_2)
        elif action == 2:
            pos_1 = self.agent_pos
            success = self.move(0,1)
            pos_2 = self.agent_pos
            composed_string = "move {} {}".format(pos_1,pos_2)
            if not success:
                pos_1 = self.agent_pos
                self.push(0,1)
                next_pos_2 = list(pos_1)
                next_pos_2[1] = str(int(next_pos_2[1]))
                next_pos_2[4] = str(int(next_pos_2[4])+2)
                next_pos_2 = "".join(next_pos_2)
                composed_string = "push {} {} {}".format(pos_1,pos_2,next_pos_2)
        elif action == 3:
            pos_1 = self.agent_pos
            success = self.move(0,-1)
            pos_2 = self.agent_pos
            composed_string = "move {} {}".format(pos_1,pos_2)
            if not success:
                pos_1 = self.agent_pos
                self.push(0,-1)
                pos_2 = self.agent_pos
                next_pos_2 = list(pos_1)
                next_pos_2[1] = str(int(next_pos_2[1]))
                next_pos_2[4] = str(int(next_pos_2[4])-2)
                next_pos_2 = "".join(next_pos_2)
                composed_string = "push {} {} {}".format(pos_1,pos_2,next_pos_2)
        else:
           raise Exception("Action out of action space") 


    
        
        return composed_string
    def checkTrap(self):
        trapped = False
        
        obj_position = self.obj_positon[0]
        
        indx_1 = int(obj_position[1])
        
        indy_1 = int(obj_position[4])

        obj_position = self.obj_positon[1]
        
        indx_2 = int(obj_position[1])
        
        indy_2 = int(obj_position[4])



        if (self.locations_init[indx_1+1][indy_1] == 0 and self.locations_init[indx_1][indy_1+1] == 0) or (self.locations_init[indx_2+1][indy_2] == 0 and self.locations_init[indx_2][indy_2+1] == 0):
            trapped = True
        elif (self.locations_init[indx_1-1][indy_1] == 0 and self.locations_init[indx_1][indy_1+1] == 0) or (self.locations_init[indx_2-1][indy_2] == 0 and self.locations_init[indx_2][indy_2+1] == 0):
            trapped = True
        elif (self.locations_init[indx_1+1][indy_1] == 0 and self.locations_init[indx_1][indy_1-1] == 0) or (self.locations_init[indx_2+1][indy_2] == 0 and self.locations_init[indx_2][indy_2-1] == 0):
            trapped = True
        elif (self.locations_init[indx_1-1][indy_1] == 0 and self.locations_init[indx_1][indy_1-1] == 0) or (self.locations_init[indx_2-1][indy_2] == 0 and self.locations_init[indx_2][indy_2-1] == 0):
            trapped = True
        elif indx_1+1 == self.grid_size-1 or indx_1-1 == 0:

            if 10 not in self.locations_init[indx_1]:
                trapped = True
        elif indy_1+1 == self.grid_size-1 or indy_1-1 == 0:

            if 10 not in self.locations_init[:,indy_1]:
                trapped = True

        elif indx_2+1 == self.grid_size-1 or indx_2-1 == 0:

            if 10 not in self.locations_init[indx_2]:
                trapped = True
        elif indy_2+1 == self.grid_size-1 or indy_2-1 == 0:

            if 10 not in self.locations_init[:,indy_2]:
                trapped = True
        
        return trapped
    
    def getGridRepresentation(self):

        grid = copy.deepcopy(self.locations_init)

        obj_position = self.obj_positon[0]
        
        indx = int(obj_position[1])
        
        indy = int(obj_position[4])
        grid[indx,indy] = 2

        obj_position = self.obj_positon[1]
        
        indx = int(obj_position[1])
        
        indy = int(obj_position[4])
        grid[indx,indy] = 2

        indx = int(self.agent_pos[1])
        indy = int(self.agent_pos[4])
        grid[indx,indy] = 5
        
        return grid
    def getStateRepresentation(self):

        obj_position = self.obj_positon[0]
        
        indx = int(obj_position[1])
        
        indy = int(obj_position[4])
        obj_positon = [indx,indy]


        obj_position2 = self.obj_positon[1]
        
        indx = int(obj_position2[1])
        
        indy = int(obj_position2[4])
        obj_positon2 = [indx,indy]


        indx = int(self.agent_pos[1])
        indy = int(self.agent_pos[4])
        agent_pos = [indx,indy]




        grid = self.locations_init.flatten()
        state = np.concatenate((agent_pos,obj_positon,obj_positon2,grid))
        
        if self.goals[0] == self.obj_positon[0] and self.goals[1] == self.obj_positon[1]:
            self.done = True
        elif self.goals[1] == self.obj_positon[0] and self.goals[0] == self.obj_positon[1]:
            self.done = True
        

        return state
   

    def reset(self):
        self.reset_state()
        self.episodes = 0
        state =  self.getStateRepresentation()
        self.done = False
        self.is_done2 = False
        self.is_done1 = False
        return state

    def reset_state(self):

        self.agent_pos = self.agent_pos_init
        self.obj_positon = copy.deepcopy(self.obj_positon_init)
        self.adjacent_points_2 = self.adjacent_points_2_init
        self.adjacent_points = self.adjacent_points_init
        self.goals = copy.deepcopy(self.goals_init)
        

       
    def get_neighbors(self):
        
        neighbors = []
        copy_env = copy.deepcopy(self)

        for action in range(4):
            if action == 0:
                copy_env = copy.deepcopy(self)
                copy_env.action_took = action
                _,reward,_,success = copy_env.step(action)
                copy_env.reward = reward
                if success:
                    neighbors.append(copy_env)           
            
            elif action == 1:
                copy_env = copy.deepcopy(self)
                copy_env.action_took = action
                _,reward,_,success = copy_env.step(action)
                copy_env.reward = reward
                if success:
                    neighbors.append(copy_env) 
            
            elif action == 2:
                copy_env = copy.deepcopy(self)
                copy_env.action_took = action
                _,reward,_,success = copy_env.step(action)
                copy_env.reward = reward
                if success:
                    neighbors.append(copy_env)    
            
            elif action == 3:
                copy_env = copy.deepcopy(self)
                copy_env.action_took = action
                _,reward,_,success = copy_env.step(action)
                copy_env.reward = reward
                if success:
                    neighbors.append(copy_env) 
            else:
                raise Exception("Action out of action space") 

        return neighbors




    def move(self, directionx,directiony):
        next_pos = list(self.agent_pos)
        
        next_pos[1] = str(int(next_pos[1])+directionx)
        next_pos[4] = str(int(next_pos[4])+directiony)
        next_pos = "".join(next_pos)
        is_adjacent = False
        
        next_pos = str(next_pos)
        if next_pos in self.obj_positon:
            return False
        for s in self.adjacent_points:
            
            if (self.agent_pos,next_pos) == s:
                is_adjacent = True

            elif (next_pos,self.agent_pos) == s:
                is_adjacent = True

        if not is_adjacent:

            return False
        
        self.agent_pos = next_pos
        
        return True
    
    def push(self, directionx,directiony):
        next_pos = list(self.agent_pos)
        next_pos_2 = list(self.agent_pos)
        
        next_pos[1] = str(int(next_pos[1])+directionx)
        next_pos[4] = str(int(next_pos[4])+directiony)
        next_pos = "".join(next_pos)

        next_pos_2[1] = str(int(next_pos_2[1])+directionx*2)
        next_pos_2[4] = str(int(next_pos_2[4])+directiony*2)
        next_pos_2 = "".join(next_pos_2)

        is_adjacent = False
        is_adjacent_2 = False
        is_adjacent_obj = False
        
        next_pos = str(next_pos)
        next_pos_2 = str(next_pos_2)
        
        if next_pos not in self.obj_positon:
            return False
        if next_pos in self.obj_positon and next_pos in self.goals:
            return False
        if next_pos_2 in self.obj_positon:
            return False
        for s in self.adjacent_points:
            if (self.agent_pos,next_pos) == s:
                is_adjacent = True
            elif (next_pos,self.agent_pos) == s:
                is_adjacent = True
        
        for s in self.adjacent_points_2:
            if (self.agent_pos,next_pos_2) == s:
                is_adjacent_2 = True
                
            elif (next_pos_2,self.agent_pos) == s:
                is_adjacent_2 = True
        
        for s in self.adjacent_points:
            if (next_pos_2,next_pos) == s:
                is_adjacent_obj = True
            elif (next_pos,next_pos_2) == s:
                is_adjacent_obj = True
        
        if not is_adjacent or not is_adjacent_2 or not is_adjacent_obj:
            return False
        
        self.agent_pos = next_pos
        for i, obj in enumerate(self.obj_positon):
            if obj == next_pos:
                self.obj_positon[i] = next_pos_2
        
        
        return True
   

    def get_heuristic(self):

        indx_a = int(self.agent_pos[1])
        indy_a = int(self.agent_pos[4])


        distance = 0
        
        dis1 = 199
        dis2 = 199
        for obj_pos in self.obj_positon:
            indx_o = int(obj_pos[1])
            indy_o = int(obj_pos[4])
            dis1 = min(dis1,abs(indx_a - indx_o) + abs(indy_a - indy_o))

            for goal in self.goals:
                indx_g = int(goal[1])
                indy_g = int(goal[4])
                dis2 = min(dis2, abs(indx_g - indx_o) + abs(indy_g - indy_o))

            distance+=dis2
        distance+=dis1
        return distance

    def initialize(self):

        for s in self.domprob.initialstate():

            s_ground = s.ground([])

            if ("ObjAt") in s_ground:
                self.obj_positon_init.append(s_ground[1])
            elif ("RobotAt") in s_ground:
                self.agent_pos_init = s_ground[1]
            elif ("Adjacent") in s_ground:
                self.adjacent_points_init.append((s_ground[1],s_ground[2]))
            elif ("Adjacent_2") in s_ground:
                self.adjacent_points_2_init.append((s_ground[1],s_ground[2]))



        
        
        for s in self.domprob.worldobjects():
         
            indx = int(s[1])
            indy = int(s[4])
            self.locations_init[indx][indy] = 1
        
        for s in self.domprob.goals():
            s_ground = s.ground([])[1]
            indx = int(s_ground[1])
            indy = int(s_ground[4])
            self.locations_init[indx][indy] = 10
            self.goals_init.append(s_ground)
        
    
    def __lt__(self, other):
        return self.action_space < other.action_space

    def __eq__(self, other): 
        if not isinstance(other, Sokoban_Env):

            return NotImplemented

        return (self.agent_pos == other.agent_pos and self.obj_positon == other.obj_positon and self.goals == other.goals)

    def get_id(self):
        id_o = self.getStateRepresentation()
        return id_o.tostring()
    
    def get_id_param(self, state):
        return state.tostring()
    
    def set_task(self,task):
        self.domainfile = '/home/ricardo/Dropbox/PlaningCode/Sokoban/domain.pddl'
        self.problemfile = '/home/ricardo/Dropbox/PlaningCode/Sokoban/pea'+str(task.task)+'.pddl'
        self.domprob = DomainProblem(self.domainfile, self.problemfile)
        self.initialize()

