import numpy as np
def pants_transformation(pent):
    transformation_list = []
    p = np.copy(pent)
    for i in range(4):
        dup = False
        for trans_p in transformation_list:
            if np.array_equal(trans_p, p):
                dup = True
                break
        if not dup :
            transformation_list.append(p)
        p = np.rot90(p)
    p = np.flip(np.copy(pent),1)
    for i in range(4):
        dup = False
        for trans_p in transformation_list:
            if np.array_equal(trans_p, p):
                dup = True
                break
        if not dup :
            transformation_list.append(p)
        p = np.rot90(p)
    return transformation_list


def get_pent_idx(pent):
    pidx = 0
    for i in range(pent.shape[0]):
        for j in range(pent.shape[1]):
            if pent[i][j] != 0:
                pidx = pent[i][j]
                break
        if pidx != 0:
            break
    return pidx
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
    idx = get_pent_idx(pent)
    m = board.shape[0]
    n = board.shape[1]
    pent_m = pent.shape[0]
    pent_n = pent.shape[1]
    for row in range(pent_m):
        for col in range(pent_n):
            if pent[row][col] != 0:
                board[x+row][y+col] = idx
    return


def area(r,c , vitisted, board, m, n):
    if ( 0 <= r < m and 0 <= c < n and (r, c) not in vitisted and board[r][c] == 0):
        vitisted.add((r,c))
    else:
        return 0
    return (1 + area(r+1, c, vitisted, board ,m, n) + area(r-1, c, vitisted, board ,m, n) +
                area(r, c-1, vitisted, board ,m, n) + area(r, c+1, vitisted, board ,m, n))

def check_board(board, m, n):
    for i in range(m - 1):
        if board[i][1] != 0 and board[i + 1][1] != 0 and board[i][0] == 0 and board[i + 1][0] == 0 :
            return False
        if board[i][n - 2] != 0 and board[i + 1][n - 2] != 0 and board[i][n - 1] == 0 and board[i + 1][n - 1] == 0 :
            return False
    for j in range(n - 1) :
        if board[1][j] != 0 and board[1][j + 1] != 0 and board[0][j] == 0 and board[0][j + 1] == 0 :
            return False
        if board[m - 2][j] != 0 and board[m - 2][j + 1] != 0 and board[m - 1][j] == 0 and board[m - 1][j + 1] == 0 :
            return False
    for i in range(m - 2):
        for j in range(n - 2) :
            if board[i + 2][j] != 0 and board[i + 2][j + 1] != 0 and board[i][j] != 0 and board[i][j + 1] != 0 and board[i + 1][j] == 0 and board[i + 1][j + 1] == 0 :
                return False
            if board[i][j + 2] != 0 and board[i + 1][j + 2] != 0 and board[i][j] != 0 and board[i + 1][j] != 0 and board[i][j + 1] == 0 and board[i + 1][j + 1] == 0 :
                return False
    return True
def recursive_fill(board, remaining_pents, pents, transform, size):
    remaining_copy = remaining_pents.copy()
    if len(remaining_pents) == 0:
        return [1]
    visited = set()
    m = board.shape[0]
    n = board.shape[1]
    for j in range(n):
        for i in range(m):
            if board[i][j] == 0 and (i,j) not in visited:
                area_size = area(i, j, visited, board, m, n)
                if area_size % size != 0:
                    return []
    single = True
    if size == 3:
        single = False
        for tri_idx in remaining_pents:
            tri = pents[tri_idx]
            if tri.shape[0] == 1 or tri.shape[1] == 1 :
                single = True
        i = m - 2
        if not single:
            if not check_board(board, m, n):
                return []
        pent_idx = remaining_copy[0]
        pent = pents[pent_idx]
        trans_list = transform[pent_idx]
        for j in range(n):
            for i in range(m):
                for pent in trans_list:
                    if fill_check(board, i, j, pent):
                        recur_board = board.copy()
                        fill(recur_board, i, j, pent)
                        recur_remain_pents = remaining_copy.copy()
                        recur_remain_pents.remove(pent_idx)
                        recur_result = recursive_fill(recur_board, recur_remain_pents, pents, transform,size)
                        if len(recur_result) != 0:
                            recur_result.append((pent, (i,j)))
                            return recur_result
        return []
    else:
        pent_idx = remaining_copy[0]
        pent = pents[pent_idx]
        tran_len = len(transform[pent_idx])
        for index in remaining_copy:
            trans_list = transform[index]
            cur_len = len(trans_list)
            if (cur_len <  tran_len):
                tran_len = cur_len
                pent_idx = index
        trans_list = transform[pent_idx]

        for i in range(m):
            for j in range(n):
                for pent in trans_list:
                    if fill_check(board, i, j, pent):
                        recur_board = board.copy()
                        fill(recur_board, i, j, pent)
                        recur_remain_pents = remaining_copy.copy()
                        recur_remain_pents.remove(pent_idx)
                        recur_result = recursive_fill(recur_board, recur_remain_pents, pents, transform, size)
                        if len(recur_result) != 0:
                            recur_result.append((pent, (i,j)))
                            return recur_result

        return []

def solve(board, pents):
    sol_board = 1 - board
    transform = {}
    for i, pentomino in enumerate(pents):
        transform[i] = pants_transformation(pentomino)
    num_pents = len(pents)
    tile = pents[0]
    tile_size = 0
    for i in range(tile.shape[0]):
        for j in range(tile.shape[1]):
            if tile[i][j] != 0:
                tile_size += 1
    remaining_pents = list(range(num_pents))
    solution = recursive_fill(sol_board, remaining_pents, pents, transform, tile_size)
    solution.pop(0)
    print ("board size:", board.shape)
    return solution

