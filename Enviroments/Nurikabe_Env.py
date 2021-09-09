from  pddlpy import DomainProblem
import numpy as np
import copy
import akro
import gym

class Nurikabe_Env(gym.Env):
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
        self.grid_size = 5
        self.grid_size_width = 5
        self.grid_size_height = 5 
        self.task = task
        self.agent_pos = ""
        self.obj_positon = []
        self.adjacent_points = []
        self.sources = {}
        self.part_of_init = {}
        self.remaining_cells_init = {}
        self.availables_init = []
        self.moving_init = False
        self.goal_init = []
        self.current_completed = []
        self.remaining_cells = []
        self.painted = []
        self.painting = ""
        self.blocked = []
        self.part_of = []
        self.availables = []
        self.group_painted = []
        self.num_goals = 2

        self.locations = []
        self.reward = 0
        self.time_step = 0
        self.agent_pos_init = ""
        self.adjacent_points_init = []
        self.goals_grid = np.zeros([self.grid_size_height,self.grid_size_width])
        
        self.locations_init = np.zeros([self.grid_size_height,self.grid_size_width])

        self.action_took = 0
        self.episodes = 0
        self.max_epidoes = 1000
        self.done_max_episodes = False
        self.is_done1 = False
        self.is_done2 = False

        self.init_counter = 0
        self.initialize()
        self.reset_state()

        self.done = False
    
    @property
    def observation_space(self):
        """gym.spaces.Box: The observation space."""
        return akro.Box(low=0,high=50,shape=(31,),dtype=int)


    @property
    def action_space(self):
        """gym.spaces.Box: The action space."""
        return akro.Discrete(6)

    def step(self,action):

        self.episodes += 1
        self.time_step+=1
        reward = 0
        success = False
        success_push = False
        if action == 0:
            success = self.move(1,0)
            if not success:
                success = self.move_painting(1,0)
        elif action == 1:
            success = self.move(-1,0)
            if not success:
                success = self.move_painting(-1,0)
        elif action == 2:
            success = self.move(0,1)
            if not success:
                success = self.move_painting(0,1)
        elif action == 3:
            success = self.move(0,-1)
            if not success:
                success = self.move_painting(0,-1)
        elif action == 4:
            success = self.start_painting()
        elif action == 5:
            success = self.end_painting()
        else:
           raise Exception("Action out of action space") 


        if success:
            reward = -1

        else:

            reward = -1
        

        counter = len(self.group_painted)
        
        if counter > self.init_counter:
            reward = 50
            self.init_counter +=1

        if len(self.group_painted) >= self.num_goals:
            reward = 1000

            self.done = True
        elif self.episodes > self.max_epidoes:
            self.done = True



        state =  self.getStateRepresentation()
        info = success
        return state, reward, self.done, info
    
    def action_verbose(self,action):
        
        
        self.episodes += 1
        self.time_step+=1
        reward = 0
        success = False
        success_push = False
        if action == 0:
            head_1 = self.agent_pos
            success = self.move(1,0)
            head_2 = self.agent_pos
            composed_string = "move {} {}".format(head_1,head_2)
            if not success:
                group_ = self.painting
                success = self.move_painting(1,0)
                head_2 = self.agent_pos
                
                remaining_2 = self.remaining_cells[group_]
                remaining_1 = self.remaining_cells[group_] + 1
                composed_string = "move-painting {} {} {} n{} n{}".format(head_1,head_2, group_, remaining_1, remaining_2)
        elif action == 1:
            head_1 = self.agent_pos
            success = self.move(-1,0)
            head_2 = self.agent_pos
            composed_string = "move {} {}".format(head_1,head_2)
            if not success:
                group_ = self.painting
                success = self.move_painting(-1,0)
                head_2 = self.agent_pos
                
                remaining_2 = self.remaining_cells[group_]
                remaining_1 = self.remaining_cells[group_] + 1
                composed_string = "move-painting {} {} {} n{} n{}".format(head_1,head_2, group_, remaining_1, remaining_2)
        elif action == 2:
            head_1 = self.agent_pos
            success = self.move(0,1)
            head_2 = self.agent_pos
            composed_string = "move {} {}".format(head_1,head_2)
            if not success:
                group_ = self.painting
                success = self.move_painting(0,1)
                head_2 = self.agent_pos
                
                remaining_2 = self.remaining_cells[group_]
                remaining_1 = self.remaining_cells[group_] + 1
                composed_string = "move-painting {} {} {} n{} n{}".format(head_1,head_2, group_, remaining_1, remaining_2)
        elif action == 3:
            head_1 = self.agent_pos
            success = self.move(0,-1)
            head_2 = self.agent_pos
            composed_string = "move {} {}".format(head_1,head_2)
            if not success:
                group_ = self.painting
                success = self.move_painting(0,-1)
                head_2 = self.agent_pos
                
                remaining_2 = self.remaining_cells[group_]
                remaining_1 = self.remaining_cells[group_] + 1
                composed_string = "move-painting {} {} {} n{} n{}".format(head_1,head_2, group_, remaining_1, remaining_2)
        elif action == 4:
            success = self.start_painting()
            head_2 = self.agent_pos
            group_ = self.painting
            remaining_2 = self.remaining_cells[group_]
            remaining_1 = self.remaining_cells[group_] + 1
            composed_string = "start-painting {} {} n{} n{}".format(head_2, group_, remaining_1, remaining_2)
        elif action == 5:
            group_ = self.painting
            success = self.end_painting()
            composed_string = "end-painting {}".format(group_)
        else:
           raise Exception("Action out of action space") 

        

        
        return composed_string
    def checkTrap(self):
        trapped = False
        
        obj_position = self.obj_positon[0]
        
        indx_1 = int(obj_position[1])
        
        indy_1 = int(obj_position[4])

        if (self.locations_init[indx_1+1][indy_1] == 0 and self.locations_init[indx_1][indy_1+1] == 0): 
            trapped = True
        elif (self.locations_init[indx_1-1][indy_1] == 0 and self.locations_init[indx_1][indy_1+1] == 0): 
            trapped = True
        elif (self.locations_init[indx_1+1][indy_1] == 0 and self.locations_init[indx_1][indy_1-1] == 0):
            trapped = True
        elif (self.locations_init[indx_1-1][indy_1] == 0 and self.locations_init[indx_1][indy_1-1] == 0):
            trapped = True
        elif indx_1+1 == self.grid_size-1 or indx_1-1 == 0:

            if 10 not in self.locations_init[indx_1]:
                trapped = True
        elif indy_1+1 == self.grid_size-1 or indy_1-1 == 0:
            if 10 not in self.locations_init[:,indy_1]:
                trapped = True


        return trapped
    
    def getGridRepresentation(self):


        empty_state = np.zeros([self.grid_size_height,self.grid_size_width])

        
        for sor in self.sources:
            addv = 0
            if sor in self.painted:
                addv = .5

            indx = int(sor[4])
            indy = int(sor[6])
            empty_state[indy][indx] = int("9" + self.sources[sor][1]) + addv
        
        for avb in self.availables:
            indx = int(avb[4])
            indy = int(avb[6])
            empty_state[indy][indx] = 5
        

        for pao in self.part_of:

            addv = 0
            if pao in self.painted:
                addv = .5
            indx = int(pao[4])
            indy = int(pao[6])
            empty_state[indy][indx] = int(self.part_of[pao])+1+10 + addv

        indx = int(self.agent_pos[4])
        indy = int(self.agent_pos[6])

        empty_state[indy][indx] = 8
        
        return empty_state
    def getStateRepresentation(self):
       
        indx = int(self.agent_pos[4])
        indy = int(self.agent_pos[6])
        agent_pos = [indx,indy]

        empty_state = np.zeros([self.grid_size_height,self.grid_size_width])

        for sor in self.sources:
            addv = 0
            if sor in self.painted:
                addv = .5

            indx = int(sor[4])
            indy = int(sor[6])
            empty_state[indy][indx] = int("9" + self.sources[sor][1]) + addv
            
        
        for avb in self.availables:
            indx = int(avb[4])
            indy = int(avb[6])
            empty_state[indy][indx] = 5
        


        for blo in self.blocked:
            indx = int(blo[4])
            indy = int(blo[6])
            empty_state[indy][indx] = -1

        for pao in self.part_of:

            addv = 0
            if pao in self.painted:
                addv = .5
            indx = int(pao[4])
            indy = int(pao[6])
            empty_state[indy][indx] = int(self.part_of[pao])+10 + addv


        remaining = np.fromiter(self.remaining_cells.values(), dtype = int)   
       
        board_state = empty_state.flatten()

        moving = int(self.moving)

        painting = 0
        if self.painting != "":
            painting = int(self.painting[1])+1

        counter = len(self.group_painted)
        goals_current = np.zeros(self.num_goals)
        for i in range(counter):
            goals_current[i] = 1

        state = np.concatenate((agent_pos,[moving],[painting],board_state,remaining,goals_current))

        return state
   

    def reset(self):
        self.init_counter = 0
        self.reset_state()
        self.episodes = 0
        state =  self.getStateRepresentation()
        self.done = False
        self.is_done2 = False
        self.is_done1 = False
        return state
  

    def reset_state(self):

        self.agent_pos = copy.deepcopy(self.agent_pos_init)
        self.moving = self.moving_init
        self.remaining_cells = copy.deepcopy(self.remaining_cells_init)
        self.part_of = copy.deepcopy(self.part_of_init)

        self.adjacent_points = self.adjacent_points_init
        self.blocked = []
        self.availables = copy.deepcopy(self.availables_init)
        

       
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




    def move(self, directionx,directiony):
        next_pos = list(self.agent_pos)

        next_pos[4] = str(int(next_pos[4])+directionx)
        next_pos[6] = str(int(next_pos[6])+directiony)
        next_pos = "".join(next_pos)
        is_adjacent = False
        
        next_pos = str(next_pos)
        
        if not self.moving:
            return False

        if next_pos in self.painted:
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
    
    def start_painting(self):

        if self.agent_pos not in self.sources:
            return False


        group_ = self.sources[self.agent_pos]

        if not self.moving or self.agent_pos not in self.sources or self.remaining_cells[group_] <= 0:
           
            return False

        if self.remaining_cells[self.sources[self.agent_pos]] <= 0:
            return False
        

        self.moving = False
        self.painting = self.sources[self.agent_pos]

        self.painted.append(self.agent_pos)
        self.remaining_cells[self.sources[self.agent_pos]] = self.remaining_cells[self.sources[self.agent_pos]] - 1

        return True

    def move_painting(self, directionx,directiony):
        next_pos = list(self.agent_pos)

        next_pos[4] = str(int(next_pos[4])+directionx)
        next_pos[6] = str(int(next_pos[6])+directiony)
        next_pos = "".join(next_pos)
        is_adjacent = False
        
        next_pos = str(next_pos)

        
        if self.moving:
            return False
        if next_pos in self.part_of:
            group_ = "g"+str(self.part_of[next_pos])

            if self.painting != group_:
                return False
            

        else:
            group_ = self.painting

        

        
        if self.remaining_cells[group_] <= 0:
            return False

        if next_pos in self.painted:
            return False

        if next_pos in self.blocked:
            return False

        for s in self.adjacent_points:

            if (self.agent_pos,next_pos) == s:
                is_adjacent = True

            elif (next_pos,self.agent_pos) == s:
                is_adjacent = True

        if not is_adjacent:
            return False
        self.agent_pos = next_pos
        self.painted.append(next_pos)

        adjacents = self.obtain_adjacents(next_pos)

        for adj in adjacents:
            if adj in self.availables:
                self.part_of[adj] = group_[1]
                self.availables.remove(adj)
            elif adj not in self.part_of:
                self.blocked.append(adj)
        self.remaining_cells[group_] = self.remaining_cells[group_] - 1
        
        return True

    def obtain_adjacents(self,agent_pos):
        
        cordinates = [[0,1],[1,0],[-1,0],[0,-1]]
        adjacent = []

        for cor in cordinates:
            is_adjacent = False
            next_pos = list(agent_pos)

            next_pos[4] = str(int(next_pos[4])+cor[0])
            next_pos[6] = str(int(next_pos[6])+cor[1])
            next_pos = "".join(next_pos)

            for s in self.adjacent_points:
                if (self.agent_pos,next_pos) == s:
                    is_adjacent = True
                elif (next_pos,self.agent_pos) == s:
                    is_adjacent = True
        
            if is_adjacent:
                adjacent.append(next_pos)
        
        return adjacent

    def end_painting(self):

        if self.painting == "":
            return False

        if self.remaining_cells[self.painting] > 0:
            return False
        self.group_painted.append(self.painting)
        self.painting = ""
        self.moving = True

        return True

        
        

    def initialize(self):
        for s in self.domprob.initialstate():
            s_ground = s.ground([])
            if ("robot-pos") in s_ground:
                self.agent_pos_init = s_ground[1]
            elif ("CONNECTED") in s_ground:
                self.adjacent_points_init.append((s_ground[1],s_ground[2]))
            elif ("SOURCE") in s_ground:
                self.sources[s_ground[1]] = s_ground[2]
            elif ("moving") in s_ground:
                self.moving_init = True
            elif ("available") in s_ground:
                self.availables_init.append(s_ground[1])
            elif ("remaining-cells") in s_ground:
                self.remaining_cells_init[s_ground[1]] = int(s_ground[2][1]) 
            elif ("part-of") in s_ground:
                self.part_of_init[s_ground[1]] = int(s_ground[2][1])  

        for s in self.domprob.goals():
            s_ground = s.ground([])
            
            if len(s_ground) > 1:
                self.goal_init.append(s_ground[1])

    
    def __lt__(self, other):
        return self.action_space < other.action_space

    def __eq__(self, other): 
        if not isinstance(other, Nurikabe_Env):

            return NotImplemented

        return (self.agent_pos == other.agent_pos and self.goal_init == other.goal_init)

    def get_id(self):
        id_o = self.getStateRepresentation()

        return np.array_str(id_o)
    
    def get_id_param(self, state):
        return state.tostring()

