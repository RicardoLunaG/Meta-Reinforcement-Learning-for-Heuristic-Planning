from pddlpy import DomainProblem
import numpy as np
import copy
import gym
import akro

class Snake_Env(gym.Env):
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
        self.task = task
        self.head_pos = ""
        self.tail_pos = ""
        self.spawn_point = ""
        self.next_snake = []
        self.blocked = []
        self.points = []
        self.spawn_points = []
        self.adjacent_points = []
        self.goals = []
        self.reward = 0
        self.time_step = 0
        self.total_reward = 0
        self.previous_state = None
        self.number_of_goals = 15

        self.head_pos_init = ""
        self.tail_pos_init = ""
        self.spawn_point_init = ""
        self.next_snake_init = []
        self.blocked_init = []
        self.points_init = []
        self.spawn_points_init = []
        self.adjacent_points_init = []
        self.goals_init = []
        self.action_took = 0
        self.episodes = 0
        self.max_epidoes = 500
        self.done_max_episodes = False
        self.initialize()
        self.reset_state()
        self.rnn_state = None
        self.value = None
        self.prevous_value = None
        self.prevous_time_step = None
        self.done = False
    @property
    def observation_space(self):
        """gym.spaces.Box: The observation space."""
        return akro.Box(low=0,high=100,shape=(71,),dtype=int)


    @property
    def action_space(self):
        """gym.spaces.Box: The action space."""
        return akro.Discrete(4)
    def step(self,action):

        self.episodes += 1
        self.time_step += 1
        reward = 0
        success = False
        success_eat_spawn = False
        success_eat = False
        if action == 0:
            success = self.move(1,0)
            if not success:
                success_eat_spawn = self.move_eat_spawn(1,0)
                if not success_eat_spawn:
                    success_eat = self.move_eat(1,0)

        elif action == 1:
            success = self.move(-1,0)
            if not success:
                success_eat_spawn = self.move_eat_spawn(-1,0)
                if not success_eat_spawn:
                    success_eat = self.move_eat(-1,0)
        elif action == 2:
            success = self.move(0,1)
            if not success:
                success_eat_spawn = self.move_eat_spawn(0,1)
                if not success_eat_spawn:
                    success_eat = self.move_eat(0,1)
        elif action == 3:
            success = self.move(0,-1)
            if not success:
                success_eat_spawn = self.move_eat_spawn(0,-1)
                if not success_eat_spawn:
                    success_eat = self.move_eat(0,-1)
        else:
           raise Exception("Action out of action space") 
  
        

        if success_eat or success_eat_spawn:
            reward = 50

            self.just_eat = True

        elif success:
            reward = -1
        else:
            reward = -1

        if len(self.goals) == 0:
            self.done = True
            reward = 1000


        state =  self.getStateRepresentation()
        info = success or success_eat or success_eat_spawn

        infos = dict(succes=info)

        return state, reward, self.done, infos
 
    def getGridRepresentation(self):

        grid = np.zeros([5,5])
        for point in self.points:
            indx = int(point[3])
            indy = int(point[5])
            grid[indy][indx] = 1
        for point in self.blocked:
            indx = int(point[3])
            indy = int(point[5])
            grid[indy][indx] = 10

        indx = int(self.head_pos[3])
        indy = int(self.head_pos[5])
        grid[indy][indx] = 5
        indx = int(self.tail_pos[3])
        indy = int(self.tail_pos[5])
        grid[indy][indx] = 4
        return grid
    def getStateRepresentation(self):

        grid = np.zeros([5,5])
        grid_blocked = np.zeros([5,5])
        for point in self.points:
            indx = int(point[3])
            indy = int(point[5])
            grid[indy][indx] = 1
        for point in self.blocked:
            indx = int(point[3])
            indy = int(point[5])
            grid_blocked[indy][indx] = 1

        indx = int(self.head_pos[3])
        indy = int(self.head_pos[5])
        headpos = [indx,indy]

        indx = int(self.tail_pos[3])
        indy = int(self.tail_pos[5])
        tailpos = [indx,indy]


        if self.spawn_point == 'dummypoint':
            indx = -10
            indy = -10
        else:

            indx = int(self.spawn_point[3])
            indy = int(self.spawn_point[5])
        next_spawn_point = [indx,indy]

        grid = grid.flatten()
        grid_blocked = grid_blocked.flatten()
        goals_one_hot = np.zeros(self.number_of_goals)
        for i in range(self.number_of_goals - len(self.goals)):
            goals_one_hot[i] = 1
        state = np.concatenate((headpos,tailpos,grid_blocked,grid,next_spawn_point,goals_one_hot))

        if len(self.goals) == 0:
            self.done = True
        elif self.episodes > self.max_epidoes:
            self.done_max_episodes = True
        return state

    def reset(self):
        self.reset_state()
        self.episodes = 0

        state =  self.getStateRepresentation()
        return state
    def reset_state(self):

        self.head_pos = self.head_pos_init
        self.tail_pos = self.tail_pos_init
        self.spawn_point = self.spawn_point_init
        self.next_snake = copy.copy(self.next_snake_init)
        self.blocked = copy.copy(self.blocked_init)
        self.points = copy.copy(self.points_init)
        self.spawn_points = copy.copy(self.spawn_points_init)
        self.adjacent_points = self.adjacent_points_init
        self.goals = copy.copy(self.goals_init)

    def action_verbose(self,action):

        self.episodes += 1
        self.time_step += 1
        reward = 0
        success = False
        success_eat_spawn = False
        success_eat = False
        composed_string = ""
        if action == 0:
            head_1 = self.head_pos
            tail_1 = self.tail_pos
            success = self.move(1,0)
            head_2 = self.head_pos
            tail_2 = self.tail_pos
            
            composed_string = "move {} {} {} {}".format(head_1,head_2,tail_1,tail_2)
            
            if not success:
                spawn_1 = self.spawn_point
                success_eat_spawn = self.move_eat_spawn(1,0)
                head_2 = self.head_pos
                spawn_2 = self.spawn_point
                composed_string = "move-and-eat-spawn {} {} {} {}".format(head_1,head_2,spawn_1,spawn_2)
                if not success_eat_spawn:
                    success_eat = self.move_eat(1,0)
                    if success_eat:
                        head_2 = self.head_pos
                        composed_string = "move-and-eat-no-spawn {} {}".format(head_1,head_2)
                    else:
                        composed_string = ""
            
            

        elif action == 1:
            head_1 = self.head_pos
            tail_1 = self.tail_pos
            success = self.move(-1,0)
            head_2 = self.head_pos
            tail_2 = self.tail_pos
            
            composed_string = "move {} {} {} {}".format(head_1,head_2,tail_1,tail_2)

            if not success:
                spawn_1 = self.spawn_point
                success_eat_spawn = self.move_eat_spawn(-1,0)
                head_2 = self.head_pos
                spawn_2 = self.spawn_point
                composed_string = "move-and-eat-spawn {} {} {} {}".format(head_1,head_2,spawn_1,spawn_2)
                if not success_eat_spawn:
                    success_eat = self.move_eat(-1,0)
                    if success_eat:
                        head_2 = self.head_pos
                        composed_string = "move-and-eat-no-spawn {} {}".format(head_1,head_2)
                    else:
                        composed_string = ""
        elif action == 2:
            head_1 = self.head_pos
            tail_1 = self.tail_pos
            success = self.move(0,1)
            head_2 = self.head_pos
            tail_2 = self.tail_pos
            composed_string = "move {} {} {} {}".format(head_1,head_2,tail_1,tail_2)
            if not success:
                spawn_1 = self.spawn_point
                success_eat_spawn = self.move_eat_spawn(0,1)
                head_2 = self.head_pos
                spawn_2 = self.spawn_point
                composed_string = "move-and-eat-spawn {} {} {} {}".format(head_1,head_2,spawn_1,spawn_2)
                if not success_eat_spawn:
                    success_eat = self.move_eat(0,1)
                    if success_eat:
                        
                        head_2 = self.head_pos
                        composed_string = "move-and-eat-no-spawn {} {}".format(head_1,head_2)
                    else:
                        composed_string = ""
                        
        elif action == 3:
            head_1 = self.head_pos
            tail_1 = self.tail_pos
            success = self.move(0,-1)
            head_2 = self.head_pos
            tail_2 = self.tail_pos
            composed_string = "move {} {} {} {}".format(head_1,head_2,tail_1,tail_2)
            if not success:
                spawn_1 = self.spawn_point
                success_eat_spawn = self.move_eat_spawn(0,-1)
                head_2 = self.head_pos
                spawn_2 = self.spawn_point
                composed_string = "move-and-eat-spawn {} {} {} {}".format(head_1,head_2,spawn_1,spawn_2)
                if not success_eat_spawn:
                    success_eat = self.move_eat(0,-1)
                    if success_eat:
                        
                        head_2 = self.head_pos
                        composed_string = "move-and-eat-no-spawn {} {}".format(head_1,head_2)
                    else:
                        composed_string = ""
        else:
           raise Exception("Action out of action space") 

  
        return composed_string


    def get_neighbors(self):
        
        neighbors = []

        for action in range(12):
            
            copy_env = copy.deepcopy(self)
            copy_env.action_took = action
            _,reward,_,success = copy_env.step(action)
            copy_env.reward = reward

            if success:
                neighbors.append(copy_env)            


        return neighbors




    def move(self, directionx,directiony):
        next_head = list(self.head_pos)
        next_head[3] = str(int(next_head[3])+directionx)
        next_head[5] = str(int(next_head[5])+directiony)
        next_head = "".join(next_head)
        is_adjacent = False
        tail = ""
        current_head = self.head_pos
        current_tal = self.tail_pos

        if next_head in self.blocked:
            return False
        if next_head in self.points:
            return False
        for s in self.adjacent_points:

            if (self.head_pos,next_head) == s:
                is_adjacent = True
            elif (next_head,self.head_pos) == s:
                is_adjacent = True
        if not is_adjacent:
            return False

        for t in self.next_snake:
            if t[1] == self.tail_pos:
                tail = t[0]
                self.next_snake.remove(t)
        
        self.blocked.append(next_head)
        self.head_pos = next_head
        self.next_snake.append((next_head,current_head))
        self.blocked.remove(current_tal)
        
        self.tail_pos = tail

        return True
    
    def move_graph(self, directionx,directiony):
        next_head = list(self.head_pos)
        next_head[3] = str(int(next_head[3])+directionx)
        next_head[5] = str(int(next_head[5])+directiony)
        next_head = "".join(next_head)
        is_adjacent = False
        tail = ""
        current_head = self.head_pos
        current_tal = self.tail_pos

        if next_head in self.blocked:
            return False
        if next_head in self.points:
            return False
        for s in self.adjacent_points:

            if (self.head_pos,next_head) == s:
                is_adjacent = True
            elif (next_head,self.head_pos) == s:
                is_adjacent = True
        if not is_adjacent:
            return False

        for t in self.next_snake:
            if t[1] == self.tail_pos:
                tail = t[0]

        
        self.blocked.append(next_head)
        self.head_pos = next_head
        self.next_snake.append((next_head,current_head))
        self.blocked.remove(current_tal)
        
        self.tail_pos = tail

        return True
  
    def move_eat_spawn(self, directionx,directiony):
        next_head = list(self.head_pos)
        next_head[3] = str(int(next_head[3])+directionx)
        next_head[5] = str(int(next_head[5])+directiony)

        next_head = "".join(next_head)

        next_spawn = ""
        is_adjacent = False
        current_head = self.head_pos

        if self.spawn_point == 'dummypoint':
            return False

        if next_head in self.blocked:

            return False

        if next_head not in self.points:

            return False

        for s in self.adjacent_points:

            if (self.head_pos,next_head) == s:
                is_adjacent = True
            elif (next_head,self.head_pos) == s:
                is_adjacent = True
        if not is_adjacent:
            return False

        for s in self.spawn_points:
            
            if s[0] == self.spawn_point:

                next_spawn = s[1]
                self.spawn_points.remove(s)

        self.blocked.append(next_head)
        self.head_pos = next_head
        
        self.next_snake.append((next_head,current_head))

        self.points.remove(next_head)
        self.points.append(self.spawn_point)

        self.spawn_point = next_spawn

        self.goals.remove(next_head)

        return True


    def move_eat(self, directionx,directiony):
        next_head = list(self.head_pos)
        next_head[3] = str(int(next_head[3])+directionx)
        next_head[5] = str(int(next_head[5])+directiony)
        
        next_head = "".join(next_head)
        is_adjacent = False
        current_head = self.head_pos
        if self.spawn_point != 'dummypoint':
            return False

        if next_head in self.blocked:

            return False

        if next_head not in self.points:

            return False

        for s in self.adjacent_points:

            if (self.head_pos,next_head) == s:
                is_adjacent = True
            elif (next_head,self.head_pos) == s:
                is_adjacent = True
        if not is_adjacent:
            return False
        
        
               
        self.blocked.append(next_head)
        self.head_pos = next_head
        
        self.next_snake.append((next_head,current_head))

        self.points.remove(next_head)
        
        self.goals.remove(next_head)

        return True
    

    def get_heuristic(self):
    
        indx = int(self.head_pos[3])
        indy = int(self.head_pos[5])

        indx_p = 0
        indy_p = 0
        distance = 0
        if len(self.points) > 0:
            for point in self.points:
                indx_p = int(point[3])
                indy_p = int(point[5])

                dis = abs(indx - indx_p) + abs(indy - indy_p)
                distance+=dis
            distance = distance/len(self.points)
        else:
            distance = 0
        return distance-(self.total_reward)

    def get_heuristic_dis(self):
    
        indx = int(self.head_pos[3])
        indy = int(self.head_pos[5])

        indx_p = 0
        indy_p = 0
        distance = 100
        if len(self.points) > 0:
            for point in self.points:
                indx_p = int(point[3])
                indy_p = int(point[5])

                dis = abs(indx - indx_p) + abs(indy - indy_p)

                distance = min(distance,dis)
           
        else:
            distance = 0
  
        return distance

    def initialize(self):
        for s in self.domprob.initialstate():
            s_ground = s.ground([])

            if ("spawn") in s_ground:
                self.spawn_point_init = s_ground[1]
            elif ("NEXTSPAWN") in s_ground:
                self.spawn_points_init.append((s_ground[1],s_ground[2]))
            elif ("ispoint") in s_ground:
                self.points_init.append(s_ground[1])
            elif ("headsnake") in s_ground:
                self.head_pos_init = s_ground[1]
            elif ("nextsnake") in s_ground:
                self.next_snake_init.append((s_ground[1],s_ground[2]))
            elif ("blocked") in s_ground:
                self.blocked_init.append(s_ground[1])
            elif ("tailsnake") in s_ground:
                self.tail_pos_init = s_ground[1]
            elif ("ISADJACENT") in s_ground:
                self.adjacent_points_init.append((s_ground[1],s_ground[2]))
        
       

        for s in self.domprob.goals():
            s_ground = s.ground([])
            
            self.goals_init.append(s_ground[1])

    
    def __lt__(self, other):
        return self.action_space < other.action_space

    def __eq__(self, other): 
        if not isinstance(other, Snake_Env):

            return NotImplemented

        return (self.head_pos == other.head_pos and self.tail_pos == other.tail_pos and self.spawn_point == other.spawn_point and 
                self.next_snake == other.next_snake and self.blocked == other.blocked and self.points == other.points and self.spawn_points == other.spawn_points
                and self.goals == other.goals)

    def get_id(self):
        id_o = self.getStateRepresentation()
        return id_o.tostring()
