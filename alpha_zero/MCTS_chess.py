#!/usr/bin/env python
import pickle
import os
import collections
import numpy as np
import math
import alpha_zero.encoder_decoder as ed
from alpha_zero.chess_board import board as c_board
import copy
import torch
import torch.multiprocessing as mp
from alpha_zero.alpha_net import ChessNet
import datetime
from hive_engine.env_hive import GamePlay
import sys
from settings import PIECE_WHITE, PIECE_BLACK
from move_checker import is_valid_move, game_is_over, \
    player_has_no_moves
from hive_engine.config import MAX_MAP_FULL

sys.setrecursionlimit(10000)
import time

class UCTNode():
    def __init__(self, game, move, parent=None):
        self.game = game # state s
        self.move = move # action index
        self.is_expanded = False
        self.parent = parent  
        self.children = {}
        self.child_priors = np.zeros([MAX_MAP_FULL*MAX_MAP_FULL*11], dtype=np.float32)
        self.child_total_value = np.zeros([MAX_MAP_FULL*MAX_MAP_FULL*11], dtype=np.float32)
        self.child_number_visits = np.zeros([MAX_MAP_FULL*MAX_MAP_FULL*11], dtype=np.float32)
        self.action_idxes = []
        self.debug = None
    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value
    
    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]
    
    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value
    
    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)
    
    def child_U(self):
        return math.sqrt(self.number_visits) * (
            abs(self.child_priors) / (1 + self.child_number_visits))
    def best_child(self):
        if len(self.action_idxes) != 0:
            bestmove = self.child_Q() + self.child_U()
            bestmove = self.action_idxes[np.argmax(bestmove[self.action_idxes])]
        else:
            bestmove = np.argmax(self.child_Q() + self.child_U())
        return bestmove
    
    def select_leaf(self):
        current = self
        while current.is_expanded:
          best_move = current.best_child()
          if type(best_move) is list:
              print(best_move)
          current = current.maybe_add_child(best_move)
        return current
    
    def add_dirichlet_noise(self,action_idxs,child_priors):
        valid_child_priors = child_priors[action_idxs] # select only legal moves entries in child_priors array
        valid_child_priors = 0.75*valid_child_priors + 0.25*np.random.dirichlet(np.zeros([len(valid_child_priors)], dtype=np.float32)+0.3)
        child_priors[action_idxs] = valid_child_priors
        return child_priors
    
    def expand(self, child_priors):
        self.is_expanded = True
        action_idxs = []; c_p = child_priors
        if len(self.game.actions()) == 0:
            self.debug = self.game
        action_idxs = self.game.actions()
        if len(action_idxs) == 0:
            self.is_expanded = False
        self.action_idxes = action_idxs
        for i in range(len(child_priors)): # mask all illegal actions
            if i not in action_idxs:
                c_p[i] = 0.0000000000
        # if self.parent.parent == None: # add dirichlet noise to child_priors in root node
        #     c_p = self.add_dirichlet_noise(action_idxs,c_p)
        self.child_priors = c_p
    
    def decode_n_move_pieces(self, board, move):
        # print(move)
        board.move(move)
        return board

    def maybe_add_child(self, move):
        if move not in self.children:
            copy_board = copy.deepcopy(self.game) # make copy of board
            # print(copy_board.actions_move)
            copy_board = self.decode_n_move_pieces(copy_board,move)
            self.children[move] = UCTNode(
              copy_board, move, parent=self)
        return self.children[move]
    
    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            if current.game.player() == 1: # same as current.parent.game.player = 0
                current.total_value += (1*value_estimate) # value estimate +1 = white win
            elif current.game.player() == 0: # same as current.parent.game.player = 1
                current.total_value += (-1*value_estimate)
            current = current.parent


class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)



def UCT_search(game_state, num_reads,net):
    root = UCTNode(game_state, move=None, parent=DummyNode())
    for i in range(num_reads):
        leaf = root.select_leaf()
        encoded_s = leaf.game.encode_board();

        encoded_s = encoded_s.transpose(2,0,1)

        encoded_s = torch.from_numpy(encoded_s).float().cuda()
        encoded_s = torch.unsqueeze(encoded_s, 0)

        child_priors, value_estimate = net(encoded_s)
        child_priors = child_priors.detach().cpu().numpy().reshape(-1); value_estimate = value_estimate.item()
        if leaf.game.game_is_over(): # if checkmate
            # print("WIN")
            leaf.backup(value_estimate); continue
        leaf.expand(child_priors) # need to make sure valid moves
        # if leaf.debug != None:
        #     return np.argmax(root.child_number_visits), root, leaf

        leaf.backup(value_estimate)
    return np.argmax(root.child_number_visits), root, None

def do_decode_n_move_pieces(board,move):
    board.move(move)
    return board

def get_policy(root):
    policy = np.zeros([MAX_MAP_FULL*MAX_MAP_FULL*11], dtype=np.float32)
    for idx in np.where(root.child_number_visits!=0)[0]:
        policy[idx] = root.child_number_visits[idx]/root.child_number_visits.sum()
    return policy

def save_as_pickle(filename, data,datapath):
    completeName = os.path.join(f"./datasets/iter{datapath}/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

def load_pickle(filename):
    completeName = os.path.join("./datasets/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data


def MCTS_self_play(chessnet,num_games,cpu,datapath):
    chessnet = chessnet
    num_games = num_games
    for idxx in range(0,num_games):
        current_board = GamePlay()
        dataset = [] # to get state, policy, value for neural network training
        # states = []
        value = 0
        while current_board.game_is_over()==False and current_board.turn() <= 100:
            # current_board.human_play()
            # current_board.get_actions(current_board.white_pieces_set)
            # len(current_board.actions())

            board_state = copy.deepcopy(current_board.encode_board())
            # encoded_s = board_state.transpose(2, 0, 1)
            # encoded_s = torch.from_numpy(encoded_s).float().cuda()
            # child_priors, value_estimate = chessnet(encoded_s)
            # root.child_number_visits[root.child_number_visits != 0]
            # c_t = time.time()
            best_move, root, leaf = UCT_search(current_board, 50, chessnet)
            # print(time.time() - c_t)
            # if leaf != None:
            #     break
            current_board = do_decode_n_move_pieces(current_board, best_move) # decode move and move piece(s)
            policy = get_policy(root)
            dataset.append([board_state, policy])
            # print(current_board.current_board,current_board.move_count); print(" ")
            if current_board.game_is_over(): # checkmate
                # print("WIN Main")
                if current_board.state.winner == PIECE_WHITE: # black wins
                    value = 1
                elif current_board.state.winner == PIECE_BLACK: # white wins
                    value = -1
        current_board.actions()
        dataset_p = []

        for idx,data in enumerate(dataset):
            s, p = data
            if idx == 0:
                dataset_p.append([s, p, 0])
            else:
                dataset_p.append([s, p, value])
        del dataset
        save_as_pickle("dataset_cpu%i_%i_%s" % (cpu,idxx, datetime.datetime.today().strftime("%Y-%m-%d")),dataset_p,datapath)


    
if __name__=="__main__":
    
    net_to_play="current_net_trained8_iter1.pth.tar"
    mp.set_start_method("spawn",force=True)
    net = ChessNet()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    net.share_memory()
    net.eval()
    print("hi")
    # torch.save({'state_dict': net.state_dict()}, os.path.join("./model_data/",\
    #                                "current_net_trained8_iter1.pth.tar"))
    #
    current_net_filename = os.path.join("./model_data/",\
                                    net_to_play)
    checkpoint = torch.load(current_net_filename)
    net.load_state_dict(checkpoint['state_dict'])


    processes = []
    for i in range(2):
        p = mp.Process(target=MCTS_self_play,args=(net,50,i,"2"))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

# current_board.state.turn -= 1
# current_board.black_pieces_set
# # current_board = leaf.debug
# # current_board.actions()
# # current_board.state.player()
# # a = current_board.get_actions(current_board.white_pieces_set)
# # current_board.next_move_tiles
# # current_board =leaf.parent.game
# # start = current_board.white_pieces_set["<class 'pieces.Beetle'>0"]
# # end = a["<class 'pieces.Beetle'>0"][0]
# # current_board.state.add_moving_piece(start.pieces[-1])
# # is_valid_move(current_board.state, start, end)
# # is_valid_move()
# # new_tile = end
# # map = np.zeros(2992)
# # map[1274] = 1
# # map = map.reshape((16,17,11))
# # poss = np.where(map == 1)
# # current_board.actions()
# #
# # action_on_board = np.zeros([16, 17, 11])
# # action_skip = np.zeros([1])
# #
# # if noTurn:
# #     action_skip = np.ones([1])
# # else:
# #     for piece, tiles in action_list.items():
# #         index = np.where(np.array(key_list) == piece)[0][0]
# #         for tile in tiles:
# #             x, y = np.where(self.board_matrix == tile)
# #             action_on_board[x[-1], y[-1], index] = 1
# # encoded_action = np.concatenate((action_on_board.reshape(-1),
# #                                  action_skip.reshape(-1)))
# #
# # encoded_action = np.where(encoded_action == 1)[0]
#
# for i in range(100):
#     count = 0
#     data_path = f"./alpha_zero/datasets/iter{i}/"
#     for idx, file in enumerate(os.listdir(data_path)):
#         datasets = []
#
#         filename = os.path.join(data_path, file)
#         with open(filename, 'rb') as fo:
#             datasets.extend(pickle.load(fo, encoding='bytes'))
#         # print(len(datasets))
#
#         if len(datasets) != 100:
#             count += 1
#
#     print(count)
            # if datasets[-1][-1] == 0:
            #     datasets[-1][-1] == -1
#         with open(filename, 'wb') as output:
#             pickle.dump(datasets, output)
#         print(datasets[-1][-1])
#     break
# len(datasets)
