import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion


def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(
                None, side, board, fro, to, single=True)
            yield [fro, to, promote]

###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.


def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [move for move in generateMoves(side, board, flags)]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(
            side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [move], {encode(*move): {}})
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.


def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''

    # Base case
    if depth == 0:
        return evaluate(board), [], {}

    moves = {}
    moveTrees = {}
    moveLists = {}

    # Go through all possible moves
    for move in generateMoves(side, board, flags):

        newside, newboard, newflags = makeMove(
            side, board, move[0], move[1], flags, move[2])

        # Fan out next level of tree
        score, moveList, moveTree = minimax(
            newside, newboard, newflags, depth - 1)

        moves[encode(*move)] = score
        moveTrees[encode(*move)] = moveTree
        moveLists[encode(*move)] = moveList

    # Choose move based on side optimization
    if len(moves) > 0:
        if side:
            best_move = min(moves, key=moves.get)
        else:
            best_move = max(moves, key=moves.get)
    else:
        return evaluate(board), [], {}

    newMoveList = []
    if moveLists[best_move] != None and len(moveLists[best_move]) >= 0:

        newMoveList = [decode(best_move), *moveLists[best_move]]

    # print(moves)
    # print(moveTrees)
    return moves[best_move], newMoveList, moveTrees


def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    # Base case
    if depth == 0:
        return evaluate(board), [], {}

    moves = {}
    moveTrees = {}
    moveLists = {}

    if side:
        value = math.inf
        # Go through all possible moves
        for move in generateMoves(side, board, flags):

            newside, newboard, newflags = makeMove(
                side, board, move[0], move[1], flags, move[2])

            # Fan out next level of tree
            score, moveList, moveTree = alphabeta(
                newside, newboard, newflags, depth - 1, alpha, beta)

            moves[encode(*move)] = score
            moveTrees[encode(*move)] = moveTree
            moveLists[encode(*move)] = moveList

            value = value if value < score else score
            beta = beta if beta < value else value

            if(beta <= alpha):
                return score, [], moveTrees

    else:
        value = -math.inf
        # Go through all possible moves
        for move in generateMoves(side, board, flags):

            newside, newboard, newflags = makeMove(
                side, board, move[0], move[1], flags, move[2])

            # Fan out next level of tree
            score, moveList, moveTree = alphabeta(
                newside, newboard, newflags, depth - 1, alpha, beta)

            moves[encode(*move)] = score
            moveTrees[encode(*move)] = moveTree
            moveLists[encode(*move)] = moveList

            value = value if value > score else score
            alpha = alpha if alpha > value else value
            if(alpha >= beta):
                return score, [], moveTrees

    # Choose move based on side optimization
    if len(moves) > 0:
        if side:
            best_move = min(moves, key=moves.get)
        else:
            best_move = max(moves, key=moves.get)
    else:
        return evaluate(board), [], {}

    newMoveList = []
    if moveLists[best_move] != None and len(moveLists[best_move]) >= 0:

        newMoveList = [decode(best_move), *moveLists[best_move]]

    # print(moves)
    # print("Move tree", moveTrees)
    return moves[best_move], newMoveList, moveTrees


def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    def findPath(side, board, flags, depth, chooser):
        # print(depth)
        if depth == 0:
            return evaluate(board), [], {}
        else:
            moves = []
            for move in generateMoves(side, board, flags):
                moves.append(move)

            if len(moves) == 0:
                return evaluate(board), [], {}

            chosen_move = chooser(moves)

            newside, newboard, newflags = makeMove(
                side, board, chosen_move[0], chosen_move[1], flags, chosen_move[2])

            score, path, tree = findPath(
                newside, newboard, newflags, depth-1, chooser)
            # print(path, [chosen_move, *path])
            return score, [chosen_move, *path], {encode(*chosen_move): tree}

    potential_moves = {}

    # Iteratively looping through all depth level moves
    for move in generateMoves(side, board, flags):
        # print(move)
        # Make move
        newside, newboard, newflags = makeMove(
            side, board, move[0], move[1], flags, move[2])

        moves = []
        for child_move in generateMoves(newside, newboard, newflags):
            moves.append(child_move)

        if len(moves) > 0:
            score = 0
            # Investigate breadth moves and keep track of the best one
            total_score = 0
            best_move = None
            best_score = best_score = math.inf if side else -math.inf
            best_move_list = None
            best_move_tree = None
            count = 0
            for i in range(breadth if len(moves) >= breadth else len(moves)):

                chosen_move = chooser(moves)
                # print("Grandchildren: ", chosen_move)
                nextMoveSide, nextMoveBoard, nextMoveFlags = makeMove(
                    side, board, chosen_move[0], chosen_move[1], flags, chosen_move[2])
                score, moveList, moveTree = findPath(
                    nextMoveSide, nextMoveBoard, nextMoveFlags, depth, chooser)

                if (side and score <= best_score) or (not(side) and score >= best_score):
                    best_score = score
                    best_move = chosen_move
                    best_move_list = moveList
                    best_move_tree = moveTree
                    # print(moveList, best_score, score)
                total_score += score
                count += 1
            # print(best_move_list, best_move)
            # Calculate the average score
            avg_score = total_score / count
            potential_moves[encode(*move)] = (avg_score, [best_move,
                                                          *best_move_list], {encode(*best_move): best_move_tree})

        else:
            potential_moves[encode(
                *move)] = (evaluate(newboard), [], {})

    if(len(potential_moves) > 0):
        best_move = None
        best_move_list = None
        best_move_tree = None
        best_score = math.inf if side else -math.inf
        for move in potential_moves:
            # print(move, potential_moves[move])
            if (side and potential_moves[move][0] <= best_score) or (not(side) and potential_moves[move][0] >= best_score):
                # print(move, potential_moves[move])
                best_score = potential_moves[move][0]
                best_move_list = potential_moves[move][1]
                best_move_tree = potential_moves[move][2]
                best_move = move
        # print([decode(best_move), *best_move_list], {best_move: best_move_tree})
        # print(best_score, [decode(best_move), *best_move_list], {best_move: best_move_tree})
        return best_score, [decode(best_move), *best_move_list], {best_move: best_move_tree}
    else:
        return evaluate(board, [], {})
