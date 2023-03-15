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
        #         current path & point
        path = path_stack.pop()
        cur_loc = path[-1]
        if cur_loc in visited:
            continue
        num_states_explored += 1
        visited.append(cur_loc)
        cur_row, cur_col = cur_loc
         #  if find the end
        if maze.isObjective(cur_row, cur_col):
            return path, num_states_explored
         #  add no_visited neighbor into new path
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



   
class MST:
    # minimum spanning tree for objectives, heuristic distance
    def __init__(self, objectives):
        self.dists = {(x, y): abs(x[0]-y[0])+abs(x[1]-y[1]) for x, y in self.pairs(objectives)}
        self.union_set = {key: {key} for key in objectives} # len()!=0 for key of the set, 0 for others
        self.roots = {key: key for key in objectives} # root for this node

    # pairs for calculate dist, no repeat
    def pairs(self, keys):
        return (pairs for tuples in (((x,y) for x in keys if x < y) for y in keys) for pairs in tuples  )  
        # pairs=set()
        # for x in keys:
        #     for y in keys:
        #         if x < y:
        #             pairs.add((x,y))
        # return pairs
    
    # Kruskal's algorithm
    def get_mst_cost(self):
        cost = 0
        # add edges with min dist
        for dist, x, y in sorted((self.dists[(x, y)], x, y) for (x, y) in self.dists):
            if self.union(x, y):
                cost += dist
        return cost

    # check if x,y have the same root (same union set)
    # if not connect them and add the distance
    # else do nothing
    def union(self, x, y):
        # if x,y not in the same union set, add them togather
        rx=self.roots[x]
        ry=self.roots[y]
        if rx == ry:
            return False
        else:
            set_x=self.union_set[rx]
            set_y=self.union_set[ry]
            merge_set=set.union(set_y,set_x)
            for key_y in self.union_set[y]:
                self.roots[key_y]=rx
                self.union_set[key_y]=merge_set
            for key_x in self.union_set[x]:
                self.roots[key_x]=rx
                self.union_set[key_x]=merge_set
            # self.union_set[rx]=set.union(set_y,set_x)
            # self.union_set[ry]=set()
            return True
        
# x,y -> dist (Manhattan distance)
# def d(x,y):
#     return abs(x[0]-y[0])+abs(x[1]-y[1])

# heuristic distance, choose min for consistent heuristic
def get_h_min(point,ends):
    h_min = sys.maxsize
    for end in ends:
        h= abs(point[0]-end[0])+abs(point[1]-end[1])
        if h < h_min:
            h_min = h
    return h_min

# path -> dist (Manhattan distance)
def g(path):
    return len(path)

# get min cost for the multi points problem by using MST
# path,maze -> cost, num_of_Objectives_not_visited, 
def get_c_min(path,maze):
    ends=maze.getObjectives() # Objectives as the end points of the path
    not_visited=[] # Objectives not visited
    for end in ends:
        if end not in path:
            not_visited.append(end)
    # if all visited
    if(len(not_visited)==0):
        return 0,0,''
    h_min = get_h_min(path[-1],not_visited)
    tree=MST(not_visited)

    return h_min+g(path)+tree.get_mst_cost(),len(not_visited),str(not_visited)

def astar(maze):
    start = maze.getStart()
    frontier = queue.PriorityQueue() # frontier, sorted queue, (Priority,Data)
    frontier.put((get_c_min([start],maze)[0], [[start]])) #(priority, path list)
    visited = {} # dict of points that has been visited
    num_of_states=0
    while not frontier.empty():
        #   current list of path, path & point
        curr_list = frontier.get()[1]
        curr_path = curr_list[0]
        curr_row, curr_col = curr_path[-1]
        curr_point = (curr_row, curr_col)
        num_of_states += 1
        
        #  if find the end
        if (get_c_min(curr_path,maze)[1]==0):
            # num_of_states=len(visited)
            # num_of_states=0
            # for sets in visited:
            #     num_of_states+=len(sets)
            return curr_path, num_of_states
        
        #  this point has been visited
        not_visited = get_c_min(curr_path,maze)[2]
        if(not_visited in visited.keys()):
            if curr_point in visited[not_visited]:
                continue
        else:
            visited[not_visited]=[]
        visited[not_visited].append(curr_point)
        # print(not_visited,"\n",visited[not_visited],"\n\n")
        # num+=len(visited[not_visited])
        
        #  add new point into new path
        for new_point in maze.getNeighbors(curr_row, curr_col):
            if new_point not in visited[not_visited]:
                new_path=curr_path+[new_point]
                frontier.put((get_c_min(new_path,maze)[0], [new_path]))
    return [], 0


