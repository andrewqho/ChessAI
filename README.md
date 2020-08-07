# ChessAI

This repository contains two different ChessAI. The first is a simple exhaustive minimax approach with depth of 4 and a rudimentary board evaluator based on piece value counting and basic positional value.

The second uses a neural network regression models (CNN and MLP) to predict Stockfish 12 evaluations of chess positions. Scores are normalized between -1 and 1. Best test MSE achieved was 0.005. However, this still resulted in subpar performance relative to the rudimentary evaluation algorithm

Next steps is to attempt a Monte Carlo Tree Search (MCTS) with a blend of simple rudimentary algorithm, the CNN, and the MLP to see if combining predictions from three sources improves performance.

Helpful resources to look at it you want to attempt something similar:
https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf
https://www.cs.utexas.edu/~arman/PlayingChessWithLimitedLookAhead_preprint.pdf
