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
    # TODO: Write your code here
    # return path, num_states_explored
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
    # TODO: Write your code here
    # return path, num_states_explored
    return [], 0


def astar(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    start = maze.getStart()
    to_visit = queue.PriorityQueue()
    to_visit.put((1, start, 0)) #(priority, (x,  y), g)
    path_tracker = {start: None}
    path = []
    visited = []
    num_states_explored = 0
    end_state = (0, 0)

    while not to_visit.empty():
        curr_state = to_visit.get()

        if curr_state[1] not in visited:

            visited.append(curr_state[1])
            num_states_explored += 1

            if maze.isObjective(curr_state[1][0], curr_state[1][1]):
                end_state = curr_state[1]
                break

            neighbors = maze.getNeighbors(curr_state[1][0], curr_state[1][1])
            for neighbor in neighbors:
                if neighbor not in visited and maze.isValidMove(neighbor[0], neighbor[1]):
                    to_visit.put((manhattan_dist(neighbor, maze) + curr_state[2] + 1, neighbor, curr_state[2] + 1))
                    path_tracker[neighbor] = curr_state[1]

    while end_state:
        path.insert(0, end_state)
        end_state = path_tracker[end_state]

    return path, num_states_explored

def manhattan_dist(pos, maze):
    objectives = maze.getObjectives()
    min_heuristic = sys.maxsize
    for objective in objectives:
        heuristic = abs(pos[0] - objective[0]) + abs(pos[1] - objective[1])
        if heuristic < min_heuristic:
            min_heuristic = heuristic

    return min_heuristic
