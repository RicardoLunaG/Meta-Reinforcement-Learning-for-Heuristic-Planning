from  pddlpy import DomainProblem
import numpy as np
import copy
import gym
import akro


class Gripper_Env(gym.Env):
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
        self.action_space_ = 12
        self.state_space = 15
        self.task = task

        self.number_rooms = 2
        self.number_balls = 4
        self.number_of_grips = 2
        self.room_descrip = ["rooma","roomb"]
        self.balls_descrip = ["ball1","ball2","ball3","ball4"]

        self.state_dictionary = {}


        self.robot_at = ""
        self.robot_at_init = ""
        self.hand_empty = False

        self.ball_locations = {}
        self.ball_locations_init = {}
        self.goals = {}
        
        self.reward = 0
        self.time_step = 0
        self.goals_init = {}


        self.action_took = 0
        self.episodes = 0
        self.max_epidoes = 200
        self.done_max_episodes = False

        self.free_grip_init = {}
        self.free_grip_init["left"] = False 
        self.free_grip_init["right"] = False 

        self.free_grip = {}

        self.carry_grip = {}
        self.carry_grip_init = {}
        self.carry_grip_init["left"] = "free"
        self.carry_grip_init["right"] = "free" 
        
        for i in range(self.number_balls):
            self.ball_locations_init["ball"+str(i+1)] = ""
        
        for i in range(self.number_balls):
            self.goals_init["ball"+str(i+1)] = ""

        self.initialize()
        self.reset_state()
        self.done = False
       
    def step(self,action):

        self.episodes += 1
        self.time_step+=1
        reward = 0
        success = False
        if action < 2: 
            success = self.move(self.room_descrip[action])
        elif action < 6:
            success = self.pick(self.balls_descrip[action-2],"right")
        elif action < 10:
            success = self.pick(self.balls_descrip[action-6],"left")
        elif action == 10:
            success = self.drop("left")
        elif action == 11:
            success = self.drop("right")
        else:
           raise Exception("Action out of action space") 

        if success:
            reward = -1     
        else:
            reward = -1

        if self.ball_locations == self.goals_init:
            reward = 400

            self.done = True
        
        elif self.episodes > self.max_epidoes:
            self.done = True

        state =  self.getStateRepresentation()
        infos = dict(succes=success)
        
        return state, reward, self.done, infos

    @property
    def observation_space(self):
        """gym.spaces.Box: The observation space."""
        return akro.Box(low=0,high=20,shape=(15,),dtype=int)


    @property
    def action_space(self):
        """gym.spaces.Box: The action space."""
        return akro.Discrete(12)

    def action_verbose(self,action):

        self.episodes += 1
        self.time_step+=1
        reward = 0
        success = False
        composed_string = ""
        if action < 2: 
            room_1 = self.robot_at
            success = self.move(self.room_descrip[action])
            room_2 = self.robot_at
            composed_string = "move {} {}".format(room_1,room_2)
        elif action < 6:
            room_1 = self.robot_at
            composed_string = "pick {} {} {}".format(self.balls_descrip[action-2],room_1,"right")
            success = self.pick(self.balls_descrip[action-2],"right")
        elif action < 10:
            room_1 = self.robot_at
            composed_string = "pick {} {} {}".format(self.balls_descrip[action-6],room_1,"left")
            success = self.pick(self.balls_descrip[action-6],"left")
        elif action == 10:
            success = self.drop("left")
            room_1 = self.robot_at
            composed_string = "drop {} {} {}".format(self.carry_grip["left"],room_1,"left")
        elif action == 11:
            success = self.drop("right")
            room_1 = self.robot_at
            composed_string = "drop {} {} {}".format(self.carry_grip["right"],room_1,"right")
        else:
           raise Exception("Action out of action space") 

       
        return composed_string
    
    def getGridRepresentation(self):
        one_hot_loctions = np.zeros(self.number_balls)

        pos = 0
        for obj in self.ball_locations.values():
            if obj != "grip":
                one_hot_loctions[pos] = self.room_descrip.index(obj)+1
            pos+=1
       
        one_hot_grip_status = np.zeros(self.number_of_grips)
        pos = 0
        for obj in self.carry_grip.values():
            if obj != "free":
                one_hot_grip_status[pos] = obj[4]
            pos+=1

        one_hot_goals = np.zeros(self.number_balls)

        pos = 0
        for obj in self.goals_init.values():
            one_hot_goals[pos] = self.room_descrip.index(obj)+1
            pos+=1
        
        agent_rooom = self.room_descrip.index(self.robot_at)
        state = np.concatenate((one_hot_loctions,one_hot_grip_status,one_hot_goals,[agent_rooom]))
        

        return state
   
    def getStateRepresentation(self):
        
        one_hot_loctions = np.zeros(self.number_balls)
 
        pos = 0
        for obj in self.ball_locations.values():
            if obj != "grip":
                one_hot_loctions[pos] = self.room_descrip.index(obj)+1
            pos+=1
       
        one_hot_grip_status = np.zeros(self.number_of_grips)
        pos = 0
        for obj in self.carry_grip.values():
            if obj != "free":
                one_hot_grip_status[pos] = obj[4]
            pos+=1

        one_hot_goals = np.zeros(self.number_balls)

        pos = 0
        for obj in self.goals_init.values():
            one_hot_goals[pos] = self.room_descrip.index(obj)+1
            pos+=1
        
        agent_rooom = self.room_descrip.index(self.robot_at)

        one_hot_goal_state = np.zeros(self.number_balls)
        counter = 0
        for goal in self.goals_init:
            
            if self.goals_init[goal] == self.ball_locations[goal]:
                one_hot_goal_state[counter] = 1
            counter += 1

        state = np.concatenate((one_hot_loctions,one_hot_grip_status,one_hot_goals,[agent_rooom],one_hot_goal_state))

        return state
   

    def reset(self):
        self.reset_state()
        self.episodes = 0
        state =  self.getStateRepresentation()
        self.done = False

        self.is_done1 = False
        return state
    
    def reset_state(self):

        self.robot_at = self.robot_at_init
        self.free_grip = copy.deepcopy(self.free_grip_init)
        self.carry_grip = copy.deepcopy(self.carry_grip_init)
        self.ball_locations = copy.deepcopy(self.ball_locations_init)

        

       
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

    def move(self, to):

        self.robot_at = to
        return True

    def pick(self, obj, gripper):
        
        agent_pos = self.robot_at


        ball_location = self.ball_locations[obj]

        if ball_location != agent_pos:
            return False

        if not self.free_grip[gripper]:
            return False 
        
        
        self.carry_grip[gripper] = obj

        self.ball_locations[obj] = "grip"
        self.free_grip[gripper] = False
        return True
    
    def drop(self, gripper):
        
        agent_pos = self.robot_at
        obj =  self.carry_grip[gripper]

        if self.free_grip[gripper]:
            return False
        
        self.ball_locations[obj] = agent_pos
        self.free_grip[gripper] = True

        self.carry_grip[gripper] = "free"
        return True
    
    

    def initialize(self):

        for s in self.domprob.initialstate():
            s_ground = s.ground([])

            if ("at") in s_ground:
                
                self.ball_locations_init[s_ground[1]] = s_ground[2]
            elif ("at-robby") in s_ground:
                
                self.robot_at_init = s_ground[1]
            elif ("free") in s_ground:
                if s_ground[1] == "right":
                    self.free_grip_init["right"] = True
                else:
                    self.free_grip_init["left"] = True
                

        
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



