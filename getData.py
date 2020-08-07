import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv 
import matplotlib.cm as cm

import chess, chess.pgn
import numpy
import sys
import os
import multiprocessing
import itertools
import random
import h5py

def read_games(filename):
    f = open(filename)

    while True:
        try:
            g = chess.pgn.read_game(f)
        except KeyboardInterrupt:
            raise
        except:
            continue

        if not g:
            break
        
        yield g

def encode(board):
    # Initialize matrix to hold encoded board
    # Each encoded board will be 8x8 with 7 channels at each square. 
    # Piece types are given as:
    # P - 1, N - 2, B - 3, R - 4, Q - 5, K - 6
    encodedBoard = np.zeros((8, 8, 6), dtype=np.int) 
    
    # Loop through each of the 64 positions on the board
    # Position 0 is where the Queen's rook initially starts
    for i in range(64):
        # Calculate row and column of square
        row = i//8
        col = i%8
        
        # Check if piece exists at specified square
        piece = board.piece_at(i)
        
        if piece is not None:
            # Get piece information
            piece_type = board.piece_at(i).piece_type
            piece_color = board.piece_at(i).color
            
            # If piece belongs to playing player, then 
            if piece_color == board.turn:
                encodedBoard[i, piece_type-1] = 1
            else:
                encodedBoard[i, piece_type-1] = -1
                
    return encodedBoard
    
def parse_game(game, game_evals, verbose=False):
    game_evals = game_evals[1].split(" ")
    
    # Holds all board states after first move (does NOT include initial board position)
    states = []
    
    # Holds evaluations for board states
    evals = []
    
    # Get board from game object
    board = game.board()
    
    # For each move
    for move, ev in zip(game.mainline_moves(), game_evals):
        try:
            # Push the move onto the board and encode it
            board.push(move)
            new_state = encode(board)

            # Add this new board state to states
            states.append(new_state)   
        
            # Add the new evaluation
            if board.turn:
                evals.append(int(ev))
            else:
                evals.append(-1 * int(ev))
            
            display()
            
        except ValueError:
            if verbose:
                print("States added: ", len(states))
                print(" Evals added: ", len(evals))
            return states, evals
    
    if verbose:
        print("States added: ", len(states))
        print(" Evals added: ", len(evals))
        
    return states, evals

def readAllGames(PGNname, CSVname, max_boards = 100000, verbose=False):
    # Read in all csv entries
    
    # Holds all stockfish evaluation scores from csv
    all_game_evals = [] 

    # reading csv file 
    with open(CSVname, 'r') as csvfile: 
        # creating a csv reader object 
        csvreader = csv.reader(csvfile) 

        # extracting field names through first row 
        fields = next(csvreader)

        # extracting each data row one by one 
        for row in csvreader: 
            all_game_evals.append(row)    
    
    # Define lists to hold board vectors and stockfish evaluations
    x_train = []
    y_train = []
    
    game_id = 1
    
    # Each game should have a corresponding game_evals row in stockfish_evals
    for game, game_evals in zip(read_games(PGNname), all_game_evals):
        if game_id % 100 == 1 and verbose:
            print("Game {}".format(game_id))
            print("Number of states so far: ", len(x_train))
        
        # Parse information for game
        states, evals = parse_game(game, game_evals)
        
        # Add each board vector and stockfish evaluation
        for b, ev in zip(states, evals):
            x_train.append(b)
            y_train.append(ev)
        
        game_id += 1
        
        # Set maximum of 500,000 board states to evaluate
        if len(x_train) > max_boards:
            # Convert to numpy array
            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train)

            # Scale all y values down to range of [-1, 1]    
            y_max = max(y_train)
            y_min = min(y_train)
            
            scalar = max(y_max, abs(y_min))
            
            y_train = y_train/scalar

            print(len(x_train), " total board states saved!")
            return x_train, y_train
    
    # Convert to numpy array
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    # Scale all y values down to range of [-1, 1]    
    y_max = max(y_train)
    y_min = min(y_train)

    scalar = max(y_max, abs(y_min))

    y_train = y_train/scalar

    print(len(x_train), " total board states saved!")
    return x_train, y_train