# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Rahul Kunji (rahulsk2@illinois.edu) on 01/16/2019

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import maze
import queue
import sys

# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
    }.get(searchMethod)(maze)


def bfs(maze):
   queue = []# frontier, fifo queue
   visited = set()# set of points that has been visited
   queue.append([maze.getStart()]) # start from begin point
   while queue:
#         current path & point
       curr_path = queue.pop(0)
       curr_row, curr_col = curr_path[-1]
       curr_point = (curr_row, curr_col)
        #  if find the end
       if maze.isObjective(curr_row, curr_col):
           return curr_path, len(visited)
#          this point has been visited
       if curr_point in visited:
           continue
       visited.add(curr_point)
        #  add new point into new path
       for new_point in maze.getNeighbors(curr_row, curr_col):
           if new_point not in visited:
               queue.append(curr_path + [new_point])
   return [], 0



def dfs(maze):
    path_stack = []
    visited = []
    num_states_explored = 0
    path_stack.append([maze.getStart()])
    while path_stack:
        path = path_stack.pop()
        cur_loc = path[-1]
        if cur_loc in visited:
            continue
        num_states_explored += 1
        visited.append(cur_loc)
        cur_row, cur_col = cur_loc
        if maze.isObjective(cur_row, cur_col):
            return path, num_states_explored
        for neighbor in maze.getNeighbors(cur_row, cur_col):
            if neighbor not in visited:
                new_path = path.copy()
                new_path.append(neighbor)
                path_stack.append(new_path)

    return path, num_states_explored


def greedy(maze):
    # Initilization

    # path: a list to record the exact path out of the given maze
    # num_states_explored: a counter to record number of points considered
    # visited: a list to record the all points considered
    # to_be_visited: a priority queue to select the choice with least costs at each step
    # path_stack: a dictionary to record the pair of points with its neighbor during the exploration
    path = []
    num_states_explored = 0
    visited = []
    to_be_visited = queue.PriorityQueue()
    start = maze.getStart()
    to_be_visited.put((1, start))
    objectives = maze.getObjectives()
    path_stack = {start: None}

    # Forward Propagation
    while not to_be_visited.empty():
        curr_state = to_be_visited.get()
        curr_pos = curr_state[1]

        # If unvisited, then explore this position
        if curr_pos not in visited:
            visited.append(curr_pos)
            num_states_explored += 1

            #  If the goal is reached, exit the loop
            if maze.isObjective(curr_pos[0], curr_pos[1]):
                end_state = curr_pos
                break

            # Find all the neighbors of current position, then explore
            for neighbor in maze.getNeighbors(curr_pos[0], curr_pos[1]):
                # For all potential unvisited neighbors, select the one with least cost (in this case manhattan distance)
                if neighbor not in visited:
                    min_heuristic = sys.maxsize
                    for objective in objectives:
                        heuristic = abs(neighbor[0] - objective[0]) + abs(neighbor[1] - objective[1])
                        if heuristic < min_heuristic:
                            min_heuristic = heuristic
                    # Put neighbors back on priority queue and sort based on their cost (distance)
                    to_be_visited.put((min_heuristic, neighbor))
                    path_stack[neighbor] = curr_pos
    # Backward Tracing   
    # From the objective position to walk backward toward initial position             
    while end_state:
        path.insert(0, end_state)
        end_state = path_stack[end_state]

    return path, num_states_explored


def astar(maze):
    start = maze.getStart()
    frontier = queue.PriorityQueue() # frontier, sorted queue, (Priority,Data)
    frontier.put((cost(maze,start,[]), [start])) #(priority, path)
    visited = set()# set of points that has been visited
    while not frontier.empty():
        #   current path & point
        curr_path = frontier.get()[1]
        curr_row, curr_col = curr_path[-1]
        curr_point = (curr_row, curr_col)
        #  if find the end
        if maze.isObjective(curr_row, curr_col):
            return curr_path, len(visited)
        #  this point has been visited
        if curr_point in visited:
            continue
        visited.add(curr_point)
        #  add new point into new path
        for new_point in maze.getNeighbors(curr_row, curr_col):
            if new_point not in visited:
                frontier.put((cost(maze,new_point,curr_path), curr_path + [new_point]))
    return [], 0

def h_dist(maze,point):
    ends = maze.getObjectives()
    h_min = sys.maxsize
    for end in ends:
        h= abs(point[0] - end[0]) + abs(point[1] - end[1])
        if h < h_min:
            h_min = h
    return h_min

def g_dist(path):
    return len(path)
    
def cost(maze,point,path):
    return h_dist(maze,point)+g_dist(path)