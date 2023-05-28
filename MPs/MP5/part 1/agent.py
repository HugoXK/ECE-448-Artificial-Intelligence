import numpy as np
import utils
import random
import math


class Agent:
    
    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        # self.s = None
        # self.a = None

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        self.reset()
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    def update_state(self, state):
        snake_head_x, snake_head_y, snake_body, food_x, food_y = state
        snake_head_grid_x = math.floor(snake_head_x/40)
        snake_head_grid_y = math.floor(snake_head_y/40)
        food_grid_x = math.floor(food_x/40)
        food_grid_y = math.floor(food_y/40)  #get index in a 12x 12 grid
        snake_body_grid_list = []
        for x, y in snake_body:
            snake_body_grid_list.append((math.floor(x/40), math.floor(y/40)))

        adjoining_wall_x = 0
        adjoining_wall_y = 0
        food_dir_x = 0
        food_dir_y = 0
        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_right = 0
        adjoining_body_left = 0
        
        if snake_head_grid_x == 1:
            adjoining_wall_x = 1
        elif snake_head_grid_x == 12:
            adjoining_wall_x = 2
        else:
            adjoining_wall_x = 0

        if snake_head_grid_y == 1:
            adjoining_wall_y = 1
        elif snake_head_grid_y == 12:
            adjoining_wall_y = 2
        else:
            adjoining_wall_y = 0

        if(food_grid_x - snake_head_grid_x) > 0:
            food_dir_x = 2
        elif(food_grid_x - snake_head_grid_x) < 0:
            food_dir_x = 1
        else:
            food_dir_x = 0
        
        if(food_grid_y - snake_head_grid_y) > 0:
            food_dir_y = 2
        elif(food_grid_y - snake_head_grid_y) < 0:
            food_dir_y = 1
        else:
            food_dir_y = 0

        if ((snake_head_grid_x, snake_head_grid_y-1) in snake_body_grid_list):
            adjoining_body_top = 1
        else:
            adjoining_body_top = 0

        if ((snake_head_grid_x, snake_head_grid_y+1) in snake_body_grid_list):
            adjoining_body_bottom = 1

        else:
            adjoining_body_bottom = 0
        if ((snake_head_grid_x-1, snake_head_grid_y) in snake_body_grid_list):
            adjoining_body_left = 1
        else:
            adjoining_body_left = 0

        if ((snake_head_grid_x+1, snake_head_grid_y) in snake_body_grid_list):
            adjoining_body_right = 1
        else:
            adjoining_body_right = 0

        
        return (adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
    
    
    def update_q_table(self, last_state, last_action, cur_state, dead, points):

        adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right = self.update_state(cur_state)
        if points - self.points > 0:
            reward = 1
        elif dead:
            reward = -1
        else:
            reward = -0.1

        upper = self.Q[adjoining_wall_x][adjoining_wall_y][food_dir_x][food_dir_y][adjoining_body_top][adjoining_body_bottom][adjoining_body_left][adjoining_body_right][0]
        bottom = self.Q[adjoining_wall_x][adjoining_wall_y][food_dir_x][food_dir_y][adjoining_body_top][adjoining_body_bottom][adjoining_body_left][adjoining_body_right][1]
        left = self.Q[adjoining_wall_x][adjoining_wall_y][food_dir_x][food_dir_y][adjoining_body_top][adjoining_body_bottom][adjoining_body_left][adjoining_body_right][2]
        right = self.Q[adjoining_wall_x][adjoining_wall_y][food_dir_x][food_dir_y][adjoining_body_top][adjoining_body_bottom][adjoining_body_left][adjoining_body_right][3]
        max_a = max(upper, bottom, left, right)

        # get last Q

        adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right = self.update_state(last_state)
        Q_val = self.Q[adjoining_wall_x][adjoining_wall_y][food_dir_x][food_dir_y][adjoining_body_top][adjoining_body_bottom][adjoining_body_left][adjoining_body_right][last_action]
        alpha = self.C / (self.C + self.N[adjoining_wall_x][adjoining_wall_y][food_dir_x][food_dir_y][adjoining_body_top][adjoining_body_bottom][adjoining_body_left][adjoining_body_right][last_action])

        ret = Q_val + alpha * (reward + self.gamma * max_a - Q_val)
        return ret
    



    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''

        if self.s != None and self.a != None and self._train:
            adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right = self.update_state(self.s)
            self.Q[adjoining_wall_x][adjoining_wall_y][food_dir_x][food_dir_y][adjoining_body_top][adjoining_body_bottom][adjoining_body_left][adjoining_body_right][self.a] = self.update_q_table(self.s, self.a, state, dead, points)
            if dead:
                self.reset()

        cur_state = state.copy()
        adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right = self.update_state(cur_state)
        U = [0, 0, 0, 0]
        for i in range(4):
            N_val = self.N[adjoining_wall_x][adjoining_wall_y][food_dir_x][food_dir_y][adjoining_body_top][adjoining_body_bottom][adjoining_body_left][adjoining_body_right][i]
            Q_val = self.Q[adjoining_wall_x][adjoining_wall_y][food_dir_x][food_dir_y][adjoining_body_top][adjoining_body_bottom][adjoining_body_left][adjoining_body_right][i]
            if N_val < self.Ne:
                U[i] = 1
            else:
                U[i] = Q_val
        action = np.argmax(U)

        self.N[adjoining_wall_x][adjoining_wall_y][food_dir_x][food_dir_y][adjoining_body_top][adjoining_body_bottom][adjoining_body_left][adjoining_body_right][action] += 1

        self.s = cur_state
        self.a = action
        self.points = points
        return action
 