# -*- coding: utf-8 -*-
import numpy as np
def pants_transformation(pent):
    transformation_list = []
    cur_pent = np.copy(pent)
    transformation_list.append(cur_pent)
    transformation_list.append(np.rot90(cur_pent, k=1))
    transformation_list.append(np.rot90(cur_pent, k=2))
    transformation_list.append(np.rot90(cur_pent, k=3))

    transformation_list.append(np.flip(cur_pent))
    transformation_list.append(np.rot90(np.flip(cur_pent), k=1))
    transformation_list.append(np.rot90(np.flip(cur_pent), k=2))
    transformation_list.append(np.rot90(np.flip(cur_pent), k=3))

    #delete duplicated items
    transformation_list_no_dup = []
    for item in transformation_list:
        dup_flag = False
        for no_dup_item in transformation_list_no_dup:
            if np.array_equal(no_dup_item, item):
                dup_flag = True
                break
        if not dup_flag:
            transformation_list_no_dup.append(item)

    return transformation_list_no_dup


def get_pent_idx_solve(pent):
    pidx = 0
    for i in range(pent.shape[0]):
        for j in range(pent.shape[1]):
            if pent[i][j] != 0:
                pidx = pent[i][j]
                break
        if pidx != 0:
            break
    if pidx == 0:
        return -1
    return pidx
def count_available(board, pent):
    m = board.shape[0]
    n = board.shape[1]
    pm = pent.shape[0]
    pn = pent.shape[1]
    count = 0
    for i in range(m-pm + 1):
        for j in range(n-pn + 1):
            this_count = 1
            for row in range(pm):
                for col in range(pn):
                    if pent[row][col] != 0:
                        if board[i+row][j+col] != 0: # Overlap
                            row = pm
                            col = pn
                            this_count = 0
                            break
            count+=this_count
    #print("count",count)
    return count
def check_put(board, x, y, pent):
    m = board.shape[0]
    n = board.shape[1]
    pm = pent.shape[0]
    pn = pent.shape[1]
    if x + pm > m or y + pn > n:
        return False
    for row in range(pm):
        for col in range(pn):
            if pent[row][col] != 0:
                if board[x+row][y+col] != 0: # Overlap
                    return False
    return True
def put(board, x, y, pent):
    number = get_pent_idx_solve(pent)
    m = board.shape[0]
    n = board.shape[1]
    pm = pent.shape[0]
    pn = pent.shape[1]
    for row in range(pm):
        for col in range(pn):
            if pent[row][col] != 0:
                board[x+row][y+col] = number
    return


def area(r,c , seen, board, m, n):
    if not ( 0 <= r < m and 0 <= c < n and (r, c) not in seen and board[r][c] == 0):
        return 0
    seen.add((r,c))
    return (1 + area(r+1, c, seen, board ,m, n) + area(r-1, c, seen, board ,m, n) +
                area(r, c-1, seen, board ,m, n) + area(r, c+1, seen, board ,m, n))

def check(board):
    seen = set()
    m = board.shape[0]
    n = board.shape[1]

    visited = np.zeros((m,n))
    result  = m*n
    for i in range(m):
        for j in range(n):

            if board[i][j] == 0 and (i,j) not in seen:
                area = area(i, j, seen, m, n)
                if area % 5 != 0:
                    return False
    return True

def can_fill(board, i, j, trans_p):
    pm = trans_p.shape[0]
    pn = trans_p.shape[1]
    m = board.shape[0]
    n = board.shape[1]
    if (i + pm > m or j + pn > n):
        return False
    else:
        #sub_board = board[i:i+pm][j:j+pn]
        for ii in range(pm):
            for jj in range(pn):
                if (board[i+ii][j+jj] == 0 and trans_p[i][j] == 0) or (board[i+ii][j+jj] != 0 and trans_p[i][j] != 0):
                    return False
    return True

def recursive(board, remaining_orig, pents, transform, size):
    if len(remaining_orig) == 0:
        return ["solution:"]

    remaining = remaining_orig.copy()

    seen = set()
    m = board.shape[0]
    n = board.shape[1]


    # visited = np.zeros((m,n))
    result  = m*n
    for j in range(n):
        for i in range(m):
            if board[i][j] == 0 and (i,j) not in seen:
                area_size = area(i, j, seen, board, m, n)
                if area_size % size != 0:
                    return []
    if size == 3:
        single = False
        for trii in remaining_orig:
            tri = pents[trii]
            if tri.shape[0] == 1 or tri.shape[1] == 1 :
                single = True
        i = m - 2

        if not single:
            i = 0
            for j in range(n - 1) :
                if board[1][j] != 0 and board[1][j + 1] != 0 and board[0][j] == 0 and board[0][j + 1] == 0 :
                    return []
            i = m - 1
            for j in range(n - 1) :
                if board[m - 2][j] != 0 and board[m - 2][j + 1] != 0 and board[i][j] == 0 and board[i][j + 1] == 0 :
                    return []
            j = 0
            for i in range(m - 1):
                if board[i][1] != 0 and board[i + 1][1] != 0 and board[i][0] == 0 and board[i + 1][0] == 0 :
                    return []
            j = n - 1
            for i in range(m - 1):
                if board[i][n - 2] != 0 and board[i + 1][n - 2] != 0 and board[i][j] == 0 and board[i + 1][j] == 0 :
                    return []
            for i in range(m - 2):
                for j in range(n - 2) :
                    if board[i + 2][j] != 0 and board[i + 2][j + 1] != 0 and board[i][j] != 0 and board[i][j + 1] != 0 and board[i + 1][j] == 0 and board[i + 1][j + 1] == 0 :
                        return []
                    if board[i][j + 2] != 0 and board[i + 1][j + 2] != 0 and board[i][j] != 0 and board[i + 1][j] != 0 and board[i][j + 1] == 0 and board[i + 1][j + 1] == 0 :
                        return []
                    
    var = pents[remaining[0]]
    var_idx = remaining[0]
    tran_len = len(transform[var_idx])
    for index in remaining:
         trans_list = transform[index]
         cur_len = len(trans_list)
         if (cur_len <  tran_len):
             tran_len = cur_len
             var_idx = index
    trans_list = transform[var_idx]

    for i in range(m):
        for j in range(n):
            for var in trans_list:
                if check_put(board, i, j, var):
                    recur_board = board.copy()
                    put(recur_board, i, j, var)
                    recur_remain = remaining.copy()
                    recur_remain.remove(var_idx)
                    recur_result = recursive(recur_board, recur_remain, pents, transform, size)
                    if len(recur_result) != 0:
                        recur_result.append((var, (i,j)))
                        return recur_result

    return []

def solve(board, pents):

    # m = board.shape[0]
    # n = board.shape[1]
    sol_board = 1 - board

    # create a dictonary to store all the transformations of each pentomino
    transform = {}
    for i, pentomino in enumerate(pents):
        transform[i] = pants_transformation(pentomino)
    # for k, v in transform.items():
    #     print(k)
    #     print(v)
    num_pents = len(pents)
    tile = pents[0]
    size = 0
    for i in range(tile.shape[0]):
        for j in range(tile.shape[1]):
            if tile[i][j] != 0:
                size += 1
    print(size)
    remaining_orig = list(range(num_pents))
    if size == 5 :
        solution = recursive(sol_board, remaining_orig, pents, transform, 5)
    elif size == 3:
        solution = recursive(sol_board, remaining_orig, pents, transform, 3)
    elif size == 2:
        solution = recursive(sol_board, remaining_orig, pents, transform, 2)
    solution.pop(0)
    # print ("solution:", solution)
    print ("board size:", board.shape)
    return solution

