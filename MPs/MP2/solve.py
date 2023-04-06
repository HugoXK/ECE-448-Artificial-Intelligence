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


def get_pent_idx(pent):
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
            flag_count = 1
            for m in range(pm):
                for n in range(pn):
                    if pent[m][n] != 0:
                        if board[i+m][j+n] != 0: # Overlap
                            m = pm
                            n = pn
                            flag_count = 0
                            break
            if flag_count:
                count += 1
    return count
def fill_check(board, x, y, pent):
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
def fill(board, x, y, pent):
    number = get_pent_idx(pent)
    m = board.shape[0]
    n = board.shape[1]
    pm = pent.shape[0]
    pn = pent.shape[1]
    for row in range(pm):
        for col in range(pn):
            if pent[row][col] != 0:
                board[x+row][y+col] = number
    return


def area(r,c , vitisted, board, m, n):
    if ( 0 <= r < m and 0 <= c < n and (r, c) not in vitisted and board[r][c] == 0):
        vitisted.add((r,c))
    else:
        return 0
    
    return (1 + area(r+1, c, vitisted, board ,m, n) + area(r-1, c, vitisted, board ,m, n) +
                area(r, c-1, vitisted, board ,m, n) + area(r, c+1, vitisted, board ,m, n))

def recursive(board, remaining_orig, pents, transform, size):
    if len(remaining_orig) == 0:
        return ["solution:"]

    remaining = remaining_orig.copy()

    visited = set()
    m = board.shape[0]
    n = board.shape[1]

    result  = m*n
    for j in range(n):
        for i in range(m):
            if board[i][j] == 0 and (i,j) not in visited:
                area_size = area(i, j, visited, board, m, n)
                if area_size % size != 0:
                    return []
    single = True
    if size == 3:
        single = False
        for tri_idx in remaining_orig:
            tri = pents[tri_idx]
            if tri.shape[0] == 1 or tri.shape[1] == 1 :
                single = True
        i = m - 2

        if not single:
            for j in range(n - 1) :
                if board[1][j] != 0 and board[1][j + 1] != 0 and board[0][j] == 0 and board[0][j + 1] == 0 :
                    return []
            for j in range(n - 1) :
                if board[m - 2][j] != 0 and board[m - 2][j + 1] != 0 and board[m - 1][j] == 0 and board[m - 1][j + 1] == 0 :
                    return []
            for i in range(m - 1):
                if board[i][1] != 0 and board[i + 1][1] != 0 and board[i][0] == 0 and board[i + 1][0] == 0 :
                    return []
            for i in range(m - 1):
                if board[i][n - 2] != 0 and board[i + 1][n - 2] != 0 and board[i][n - 1] == 0 and board[i + 1][n - 1] == 0 :
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
                if fill_check(board, i, j, var):
                    recur_board = board.copy()
                    fill(recur_board, i, j, var)
                    recur_remain = remaining.copy()
                    recur_remain.remove(var_idx)
                    recur_result = recursive(recur_board, recur_remain, pents, transform, size)
                    if len(recur_result) != 0:
                        recur_result.append((var, (i,j)))
                        return recur_result

    return []

def solve(board, pents):
    sol_board = 1 - board

    # create a dictonary to store all the transformations of each pentomino
    transform = {}
    for i, pentomino in enumerate(pents):
        transform[i] = pants_transformation(pentomino)

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
    print ("board size:", board.shape)
    return solution

