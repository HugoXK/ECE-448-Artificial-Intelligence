from time import sleep
from math import inf
from random import randint
import time


class ultimateTicTacToe:
    def __init__(self):
        """
        Initialization of the game.
        """
        self.board = [['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_']]
        self.maxPlayer = 'X'
        self.minPlayer = 'O'
        self.maxDepth = 3
        # The start indexes of each local board
        self.globalIdx = [(0, 0), (0, 3), (0, 6), (3, 0),
                          (3, 3), (3, 6), (6, 0), (6, 3), (6, 6)]

        # Start local board index for reflex agent playing
        self.startBoardIdx = 4
        #self.startBoardIdx = randint(0, 8)

        # utility value for reflex offensive and reflex defensive agents
        self.winnerMaxUtility = 10000
        self.twoInARowMaxUtility = 500
        self.preventThreeInARowMaxUtility = 100
        self.cornerMaxUtility = 30

        self.winnerMinUtility = -10000
        self.twoInARowMinUtility = -100
        self.preventThreeInARowMinUtility = -500
        self.cornerMinUtility = -30

        self.expandedNodes = 0
        self.currPlayer = True
            
    def getNextBoardIdx(self, i, j):
        return (i % 3) * 3 + j % 3

    def printGameBoard(self):
        """
        This function prints the current game board.
        """
        print('\n'.join([' '.join([str(cell) for cell in row])
                         for row in self.board[:3]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row])
                         for row in self.board[3:6]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row])
                         for row in self.board[6:9]])+'\n')

    def evaluatePredifined(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for predefined agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                    True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        
        # 1st
        if self.checkWinner() == 1 and isMax:  # offensive
            return self.winnerMaxUtility
        elif self.checkWinner() == -1 and not isMax:  # defensive
            return self.winnerMinUtility

        # 2nd
        score = 0
        count_unblocked = 0  # unblocked 2 in row
        count_prevention = 0  # prevent 2 in row
        count_2=0 # count for unblocked 2 in row and prevention, sum of both cases
        
        if isMax:
            curr_player = self.maxPlayer
            oppo_player = self.minPlayer
        else:
            curr_player = self.minPlayer
            oppo_player = self.maxPlayer
            
        for i in range(9):
            row, col = self.globalIdx[i]
            # Rows
            for j in range(3):
                rows = [self.board[row+j][col], self.board[row+j][col+1], self.board[row+j][col+2]]
                if rows.count(curr_player) == 2 :
                    count_2 += 1 # count for 2 in the same line
                    count_prevention += rows.count(oppo_player) # if 1, prevent, increment
            # Columns
            # for j in range(3):
                cols = [self.board[row][col+j], self.board[row+1][col+j], self.board[row+2][col+j]]
                if cols.count(curr_player) == 2 :
                    count_2 += 1
                    count_prevention += cols.count(oppo_player)

            # Diagonals
            diagonal = [self.board[row][col], self.board[row+1][col+1], self.board[row+2][col+2]]
            if diagonal.count(curr_player) == 2 :
                count_2 += 1
                count_prevention += diagonal.count(oppo_player)
            
            diagonal = [self.board[row+2][col], self.board[row+1][col+1], self.board[row][col+2]]
            if diagonal.count(curr_player) == 2 :
                count_2 += 1
                count_prevention += diagonal.count(oppo_player)

        count_unblocked = count_2 - count_prevention
        if isMax:
            score += self.twoInARowMaxUtility * count_unblocked + self.preventThreeInARowMaxUtility * count_prevention
        else:
            score += self.twoInARowMinUtility * count_unblocked + self.preventThreeInARowMinUtility * count_prevention

        # 3rd
        if score == 0:
            for i in range(9):
                row, col = self.globalIdx[i]
                for y, x in [(row, col), (row+2, col), (row, col+2), (row+2, col+2)]:
                    if self.board[y][x] == self.maxPlayer and isMax:
                        score += self.cornerMaxUtility
                    elif self.board[y][x] == self.minPlayer and not isMax:
                        score += self.cornerMinUtility
                        
        return score

    def evaluateDesigned(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for your own agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        #YOUR CODE HERE
        score=0
        # 1) First Rule: If the defensive agent wins (forms three-in-a-row), set the utility score to be - 10000.
        # 2) Second Rule: For each unblocked two-in-a-row, increment the utility score by 500; 
        #              For each prevention, increment the utility score by 100.
        # 3) Third Rule: For each corner taken by defensive agent, decrement the utility score by 30.
        # 1st
        if self.checkWinner() == 1 and isMax:  # defensive
            return self.winnerMinUtility
        elif self.checkWinner() == -1 and not isMax:  # offensive
            return self.winnerMaxUtility

        # 2nd
        score = 0
        count_unblocked = 0  # unblocked 2 in row
        count_prevention = 0  # prevent 2 in row
        count_2 = 0 # count for unblocked 2 in row and prevention, sum of both cases
        
        if isMax:
            curr_player = self.minPlayer
            oppo_player = self.maxPlayer
        else:
            curr_player = self.maxPlayer
            oppo_player = self.minPlayer       

        for i in range(9):
            row, col = self.globalIdx[i]
            # Rows
            for j in range(3):
                rows = [self.board[row+j][col], self.board[row+j][col+1], self.board[row+j][col+2]]
                if rows.count(curr_player) == 2 :
                    count_2 += 1 # count for 2 in the same line
                    count_prevention += rows.count(oppo_player) # if 1, prevent, increment
            # Columns
            # for j in range(3):
                cols = [self.board[row][col+j], self.board[row+1][col+j], self.board[row+2][col+j]]
                if cols.count(curr_player) == 2 :
                    count_2 += 1
                    count_prevention += cols.count(oppo_player)

            # Diagonals
            diagonal = [self.board[row][col], self.board[row+1][col+1], self.board[row+2][col+2]]
            if diagonal.count(curr_player) == 2 :
                count_2 += 1
                count_prevention += diagonal.count(oppo_player)
            
            diagonal = [self.board[row+2][col], self.board[row+1][col+1], self.board[row][col+2]]
            if diagonal.count(curr_player) == 2 :
                count_2 += 1
                count_prevention += diagonal.count(oppo_player)

        count_unblocked = count_2 - count_prevention
        if isMax:
            score += self.twoInARowMinUtility * count_unblocked + self.preventThreeInARowMinUtility * count_prevention
        else:
            score += self.twoInARowMaxUtility * count_unblocked + self.preventThreeInARowMaxUtility * count_prevention

        # 3rd
        if score == 0:
            for i in range(9):
                row, col = self.globalIdx[i]
                for y, x in [(row, col), (row+2, col), (row, col+2), (row+2, col+2)]:
                    if self.board[y][x] == self.maxPlayer and isMax:
                        score += self.cornerMinUtility
                    elif self.board[y][x] == self.minPlayer and not isMax:
                        score += self.cornerMaxUtility
        return score

    def checkMovesLeft(self):
        """
        This function checks whether any legal move remains on the board.
        output:
        movesLeft(bool): boolean variable indicates whether any legal move remains
                        on the board.
        """
        for i, j in self.globalIdx:
            if any(self.board[k][l] == '_' for k in range(i, i+3) for l in range(j, j+3)):
                return True
        return False

    def checkWinner(self):
        """
        This function checks whether there is a winner on the board.
        output:
        winner(int): Return 0 if there is no winner.
                     Return 1 if maxPlayer is the winner.
                     Return -1 if miniPlayer is the winner.
        """
        for i in range(9):
            row, col = self.globalIdx[i]
            # Rows
            for j in range(3):
                rows = [self.board[row+j][col], self.board[row+j][col+1], self.board[row+j][col+2]]
                if rows.count(self.maxPlayer)==3:
                    return 1
                elif rows.count(self.minPlayer)==3:
                    return -1
            # Columns
            # for j in range(3):
                cols = [self.board[row][col+j], self.board[row+1][col+j], self.board[row+2][col+j]]
                if cols.count(self.maxPlayer)==3:
                    return 1
                elif cols.count(self.minPlayer)==3:
                    return -1

            # Diagonals
            diagonal = [self.board[row][col], self.board[row+1][col+1], self.board[row+2][col+2]]
            if diagonal.count(self.maxPlayer)==3:
                return 1
            elif diagonal.count(self.minPlayer)==3:
                return -1
            
            diagonal = [self.board[row+2][col], self.board[row+1][col+1], self.board[row][col+2]]
            if diagonal.count(self.maxPlayer)==3:
                return 1
            elif diagonal.count(self.minPlayer)==3:
                return -1
        return 0

    def alphabeta(self,depth,currBoardIdx,alpha,beta,isMax):
        """
        This function implements alpha-beta algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        best_value(float):the best_value that current player may have
        """
        #YOUR CODE HERE
        self.expandedNodes += 1
        if (depth == self.maxDepth) or (self.checkMovesLeft() == 0) or (self.checkWinner() != 0): 
            return self.evaluatePredifined(not isMax)

        # MaxPlayer  
        if isMax:
            bestValue = -self.winnerMaxUtility
            possibleMoveOptions = self.possibleMoves(currBoardIdx)
            for moveOption in possibleMoveOptions:
                self.board[moveOption[0]][moveOption[1]] = self.maxPlayer
                bestValue = max(bestValue, self.alphabeta(depth+1, (moveOption[0]%3)*3 + moveOption[1]%3, alpha, beta, not isMax))
                self.board[moveOption[0]][moveOption[1]] = '_'
                if (bestValue >= beta): 
                    return bestValue
                alpha = max(bestValue, alpha)
        # MinPlayer
        else: 
            bestValue = -self.winnerMinUtility
            possibleMoveOptions = self.possibleMoves(currBoardIdx)
            for moveOption in possibleMoveOptions:
                self.board[moveOption[0]][moveOption[1]] = self.minPlayer
                bestValue = min(bestValue, self.alphabeta(depth+1, (moveOption[0]%3)*3 + moveOption[1]%3, alpha, beta, not isMax))
                self.board[moveOption[0]][moveOption[1]] = '_'
                if (bestValue <= alpha): 
                    return bestValue
                beta = min(bestValue, beta)

        return 

    def minimax(self, depth, currBoardIdx, isMax):
        """
        This function implements minimax algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        # YOUR CODE HERE
        if (not self.checkMovesLeft()) or (self.checkWinner() != 0) or (depth == self.maxDepth) :
            self.expandedNodes += 1
            return self.evaluatePredifined(self.currPlayer)

        # init
        if isMax:
            curr_player = self.maxPlayer
            best_value = self.winnerMinUtility
        else:
            curr_player = self.minPlayer
            best_value = self.winnerMaxUtility
            
        # suppose fill in one box and call minimax for deeper
        row, col = self.globalIdx[currBoardIdx]
        for i in range(row,row+3):
            for j in range(col,col+3):
                if self.board[i][j] == '_':
                    self.board[i][j] = curr_player
                    curr_value = self.minimax(depth+1, self.getNextBoardIdx(i, j), not isMax)
                    self.board[i][j] = '_'
                    if isMax:
                        best_value = max(best_value, curr_value)
                    else:
                        best_value = min(best_value, curr_value)    
         
        return best_value
     


    def playGamePredifinedAgent(self, maxFirst, isMinimaxOffensive, isMinimaxDefensive):
        """
        This function implements the processes of the game of predifined offensive agent vs defensive agent.
        input args:
        maxFirst(bool): boolean variable indicates whether maxPlayer or minPlayer plays first.
                        True for maxPlayer plays first, and False for minPlayer plays first.
        isMinimaxOffensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for offensive agent.
                        True is minimax and False is alpha-beta.
        isMinimaxOffensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for defensive agent.
                        True is minimax and False is alpha-beta.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        bestValue(list of float): list of bestValue at each move
        expandedNodes(list of int): list of expanded nodes at each move
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        # YOUR CODE HERE
        self.currPlayer = maxFirst
        currBoardIdx = self.startBoardIdx
        # self.expandedNodes = 0
        
        bestMove = []
        bestValue=[]
        expandedNodes=[]
        gameBoards=[]
        winner = 0
        
        alpha = self.winnerMinUtility
        beta = self.winnerMaxUtility
        
        best_move=(0,0)# not used
        
        while self.checkMovesLeft() and self.checkWinner() == 0:
            if self.currPlayer:
                curr_player = self.maxPlayer
                best_value = self.winnerMinUtility
                row, col = self.globalIdx[currBoardIdx]
                for i in range(row,row+3):
                    for j in range(col,col+3):
                        if self.board[i][j] == '_':
                            self.board[i][j] = curr_player
                            if isMinimaxOffensive:
                                curr_value = self.minimax(1, self.getNextBoardIdx(i, j), not self.currPlayer)
                            else:
                                curr_value = self.alphabeta(1, self.getNextBoardIdx(i, j), alpha, beta, not self.currPlayer)
                            self.board[i][j] = '_'
                        
                            if curr_value > best_value:
                                best_move = (i, j)
                                best_value = curr_value
                            
            else:
                curr_player = self.minPlayer
                best_value = self.winnerMaxUtility
                row, col = self.globalIdx[currBoardIdx]
                for i in range(row,row+3):
                    for j in range(col,col+3):
                        if self.board[i][j] == '_':
                            self.board[i][j] = curr_player
                            if isMinimaxDefensive:
                                curr_value = self.minimax(1, self.getNextBoardIdx(i, j), not self.currPlayer)
                            else:
                                curr_value = self.alphabeta(1, self.getNextBoardIdx(i, j), alpha, beta, not self.currPlayer)
                            self.board[i][j] = '_'
                            
                            if curr_value < best_value:
                                best_move = (i, j)
                                best_value = curr_value
                                
            self.board[best_move[0]][best_move[1]] = curr_player
            currBoardIdx = self.getNextBoardIdx(best_move[0], best_move[1])
            bestMove.append(best_move)
            bestValue.append(best_value)
            expandedNodes.append(self.expandedNodes)
            gameBoards.append(self.board)
            self.printGameBoard()
            self.currPlayer = not self.currPlayer
            

        winner = self.checkWinner()
        return gameBoards, bestMove, expandedNodes, bestValue, winner

    def playGameYourAgent(self,maxFirst,isMinimaxOffensive,isMinimaxDefensive):
        """
        This function implements the processes of the game of your own agent vs predifined offensive agent.
        input args:
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        self.currPlayer = maxFirst
        currBoardIdx = self.startBoardIdx
        # self.expandedNodes = 0
        
        bestMove = []
        bestValue=[]
        expandedNodes=[]
        gameBoards=[]
        winner = 0
        
        alpha = self.winnerMinUtility
        beta = self.winnerMaxUtility
        
        best_move=(0,0)# not used
        
        while self.checkMovesLeft() and self.checkWinner() == 0:
            if self.currPlayer:
                curr_player = self.maxPlayer
                best_value = self.winnerMinUtility
                row, col = self.globalIdx[currBoardIdx]
                for i in range(row,row+3):
                    for j in range(col,col+3):
                        if self.board[i][j] == '_':
                            self.board[i][j] = curr_player
                            if isMinimaxOffensive:
                                curr_value = self.minimax(1, self.getNextBoardIdx(i, j), not self.currPlayer)
                            else:
                                curr_value = self.alphabeta(1, self.getNextBoardIdx(i, j), alpha, beta, not self.currPlayer)
                            self.board[i][j] = '_'
                        
                            if curr_value > best_value:
                                best_move = (i, j)
                                best_value = curr_value
                            
            else:
                curr_player = self.minPlayer
                best_value = self.winnerMaxUtility
                row, col = self.globalIdx[currBoardIdx]
                for i in range(row,row+3):
                    for j in range(col,col+3):
                        if self.board[i][j] == '_':
                            self.board[i][j] = curr_player
                            if isMinimaxDefensive:
                                curr_value = self.minimax(1, self.getNextBoardIdx(i, j), not self.currPlayer)
                            else:
                                curr_value = self.alphabeta(1, self.getNextBoardIdx(i, j), alpha, beta, not self.currPlayer)
                            self.board[i][j] = '_'
                            
                            if curr_value < best_value:
                                best_move = (i, j)
                                best_value = curr_value
                                
            self.board[best_move[0]][best_move[1]] = curr_player
            currBoardIdx = self.getNextBoardIdx(best_move[0], best_move[1])
            bestMove.append(best_move)
            bestValue.append(best_value)
            expandedNodes.append(self.expandedNodes)
            gameBoards.append(self.board)
            self.printGameBoard()
            self.currPlayer = not self.currPlayer
            

        winner = self.checkWinner()
        return gameBoards, bestMove, winner


    def playGameHuman(self,maxFirst,isHumanOffensive,isAgentMinimax):
        """
        This function implements the processes of the game of your own agent vs a human.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        bestMove=[]
        gameBoards=[]
        winner=0
        expandedNodes=[]

        self.currPlayer = maxFirst
        currentBoardIdx = self.startBoardIdx
        alpha = -self.winnerMaxUtility 
        beta = -self.winnerMinUtility
        move = 0

        if isHumanOffensive:
            while self.checkMovesLeft() and self.checkWinner() == 0:
                if self.currPlayer:
                    curr_player = self.maxPlayer
                    self.printGameBoard()
                    print("Current local board has index", currentBoardIdx)
                    X = int(input("Enter X (0, 1, 2) coordinate:")) 
                    Y = int(input("Enter Y (0, 1, 2) coordinate:"))
                    while((X not in self.validSet) or (Y not in self.validSet) or (self.board[currentBoard[0]+X][currentBoard[1]+Y] != '_')):
                        X = int(input("Wrong, enter X (0, 1, 2) coordinate:")) 
                        Y = int(input("Enter Y (0, 1, 2) coordinate:"))
                    bestMove = (currentBoard[0]+X, currentBoard[1]+Y)
                                
                else:
                    curr_player = self.minPlayer
                    best_value = self.winnerMinUtility
                    row, col = self.globalIdx[currBoardIdx]
                    for i in range(row,row+3):
                        for j in range(col,col+3):
                            if self.board[i][j] == '_':
                                self.board[i][j] = curr_player
                                if isAgentMinimax:
                                    curr_value = self.minimax(1, self.getNextBoardIdx(i, j), not self.currPlayer)
                                else:
                                    curr_value = self.alphabeta(1, self.getNextBoardIdx(i, j), alpha, beta, not self.currPlayer)
                                self.board[i][j] = '_'                      
                                if curr_value > best_value:
                                    best_move = (i, j)
                                    best_value = curr_value
                                    
                self.board[best_move[0]][best_move[1]] = curr_player
                currBoardIdx = self.getNextBoardIdx(best_move[0], best_move[1])
                bestMove.append(best_move)
                bestValue.append(best_value)
                expandedNodes.append(self.expandedNodes)
                gameBoards.append(self.board)
                self.printGameBoard()
                self.currPlayer = not self.currPlayer
            
            winner = self.checkWinner()

        else:

            while self.checkMovesLeft() and self.checkWinner() == 0:
                if self.currPlayer:
                    curr_player = self.minPlayer
                    best_value = self.winnerMinUtility
                    row, col = self.globalIdx[currBoardIdx]
                    for i in range(row,row+3):
                        for j in range(col,col+3):
                            if self.board[i][j] == '_':
                                self.board[i][j] = curr_player
                                if isAgentMinimax:
                                    curr_value = self.minimax(1, self.getNextBoardIdx(i, j), not self.currPlayer)
                                else:
                                    curr_value = self.alphabeta(1, self.getNextBoardIdx(i, j), alpha, beta, not self.currPlayer)
                                self.board[i][j] = '_'                      
                                if curr_value > best_value:
                                    best_move = (i, j)
                                    best_value = curr_value
                                
                else:
                    curr_player = self.maxPlayer
                    self.printGameBoard()
                    print("Current local board has index", currentBoardIdx)
                    X = int(input("Enter X (0, 1, 2) coordinate:")) 
                    Y = int(input("Enter Y (0, 1, 2) coordinate:"))
                    while((X not in self.validSet) or (Y not in self.validSet) or (self.board[currentBoard[0]+X][currentBoard[1]+Y] != '_')):
                        X = int(input("Wrong, enter X (0, 1, 2) coordinate:")) 
                        Y = int(input("Enter Y (0, 1, 2) coordinate:"))
                    bestMove = (currentBoard[0]+X, currentBoard[1]+Y)
                                    
                self.board[best_move[0]][best_move[1]] = curr_player
                currBoardIdx = self.getNextBoardIdx(best_move[0], best_move[1])
                bestMove.append(best_move)
                bestValue.append(best_value)
                expandedNodes.append(self.expandedNodes)
                gameBoards.append(self.board)
                self.printGameBoard()
                self.currPlayer = not self.currPlayer
            
            winner = self.checkWinner()

        return gameBoards, bestMove, winner

    
if __name__ == "__main__":
    uttt = ultimateTicTacToe()

    start = time.time()
    # gameBoards, bestMove, expandedNodes, bestValue, winner = uttt.playGamePredifinedAgent(True, False, False)
    gameBoards, bestMove, expandedNodes, bestValue, winner = uttt.playGamePredifinedAgent(True, True, True)

    print("time spent: ", time.time() - start)

    if winner == 1:
        print("The winner is maxPlayer!!!")
    elif winner == -1:
        print("The winner is minPlayer!!!")
    else:
        print("Tie. No winner:(")
