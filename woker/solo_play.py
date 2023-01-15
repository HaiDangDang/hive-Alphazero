"""
This encapsulates all of the functionality related to actually playing the game itself, not just
making / training predictions.
"""
import torch
import os
from copy import deepcopy
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from threading import Lock

import numpy as np
from hive_engine.env_hive import GamePlay

from hive_engine.config import ACTION_SPACE

from settings import PIECE_WHITE, PIECE_BLACK

from hive_engine.config import SEARCH_THREADS, MAX_GAME_LENGTH, MAX_MAP_FULL


simulation_num_per_move = 100
tau_decay_rate = 0.01

c_puct = 0.7

dirichlet_alpha = 0.3
noise_eps = 0.25
virtual_loss = 1

# these are from AGZ nature paper
class VisitStats:
    """
    Holds information for use by the AGZ MCTS algorithm on all moves from a given game state (this is generally used inside
    of a defaultdict where a game state in FEN format maps to a VisitStats object).
    Attributes:
        :ivar defaultdict(ActionStats) a: known stats for all actions to take from the the state represented by
            this visitstats.
        :ivar int sum_n: sum of the n value for each of the actions in self.a, representing total
            visits over all actions in self.a.
    """
    def __init__(self):
        self.a = defaultdict(ActionStats)
        self.sum_n = 0
        self.actions = []

class ActionStats:
    """
    Holds the stats needed for the AGZ MCTS algorithm for a specific action taken from a specific state.
    Attributes:
        :ivar int n: number of visits to this action by the algorithm
        :ivar float w: every time a child of this action is visited by the algorithm,
            this accumulates the value (calculated from the value network) of that child. This is modified
            by a virtual loss which encourages threads to explore different nodes.
        :ivar float q: mean action value (total value from all visits to actions
            AFTER this action, divided by the total number of visits to this action)
            i.e. it's just w / n.
        :ivar float p: prior probability of taking this action, given
            by the policy network.
    """
    def __init__(self):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0


class HivePlayer:
    """
    Plays the actual game of chess, choosing moves based on policy and value network predictions coming
    from a learned model on the other side of a pipe.
    Attributes:
        :ivar list: stores info on the moves that have been performed during the game
        :ivar Config config: stores the whole config for how to run
        :ivar PlayConfig play_config: just stores the PlayConfig to use to play the game. Taken from the config
            if not specifically specified.
        :ivar int labels_n: length of self.labels.
        :ivar list(str) labels: all of the possible move labels (like a1b1, a1c1, etc...)
        :ivar dict(str,int) move_lookup: dict from move label to its index in self.labels
        :ivar list(Connection) pipe_pool: the pipes to send the observations of the game to to get back
            value and policy predictions from
        :ivar dict(str,Lock) node_lock: dict from FEN game state to a Lock, indicating
            whether that state is currently being explored by another thread.
        :ivar VisitStats tree: holds all of the visited game states and actions
            during the running of the AGZ algorithm
    """
    # dot = False
    def __init__(self, pipes=None, reward=False):
        self.moves = []

        self.tree = defaultdict(VisitStats)

        self.pipe_pool = pipes
        self.node_lock = defaultdict(Lock)
        self.none_queue = True
        self.net = None
        self.simulation_num_per_move = simulation_num_per_move
        self.reward = reward
        self.main_key_state = None
        self.max_depth = None

    def reset(self):
        """
        reset the tree to begin a new exploration of states
        """
        self.tree = defaultdict(VisitStats)


    def action(self, env, non_queue=True) -> str:

        self.reset()
        self.max_depth = env.state.turn
        # for tl in range(self.play_config.thinking_loop):
        self.main_key_state = env.state_key
        root_value, naked_value = self.search_moves(env)
        policy, sum_all = self.calc_policy(env)
        p = self.apply_temperature(policy,  int(env.state.turn+1)/2)
        my_action = int(np.random.choice(range(ACTION_SPACE), p=p))

        state = env.state_key
        my_visitstats = self.tree[state]
        w = np.zeros(ACTION_SPACE)
        for action, a_s in my_visitstats.a.items():
            w[action] = a_s.w
        as_ss = np.flip(np.argsort(w))[:3]
        key_list = list(env.white_pieces_set.keys())
        for a in as_ss:
            map = np.zeros(MAX_MAP_FULL * MAX_MAP_FULL * 11)
            map[a] = 1
            map = map.reshape((MAX_MAP_FULL, MAX_MAP_FULL, 11))
            poss = np.where(map == 1)
            end_tile = env.board_matrix[poss[0][-1], poss[1][-1]]
            piece = key_list[poss[2][-1]]
            print(piece, end_tile.core_index, w[a])


        print("MAX DEPTH ", self.max_depth)
        print(env.decode_action(my_action), env.state.turn)

        #
        # my_action = np.argmax(policy)
        # if can_stop and self.play_config.resign_threshold is not None and \
        #                 root_value <= self.play_config.resign_threshold \
        #                 and env.num_halfmoves > self.play_config.min_resign_turn:
        #     # noinspection PyTypeChecker
        #     return None
        # else:
        # self.moves.append([env.observation, list(policy)])

        return my_action, [list(policy), sum_all]

    def search_moves(self, env) -> (float, float):
        if self.none_queue:
            futures = []
            with ThreadPoolExecutor(max_workers=SEARCH_THREADS) as executor:
                for _ in range(self.simulation_num_per_move):
                    futures.append(executor.submit(self.search_my_move, deepcopy(env), is_root_node=True))

            vals = [f.result() for f in futures]
        else:
            vals = [self.search_my_move(deepcopy(env), is_root_node=True) for _ in
                    range(self.simulation_num_per_move)]

        return np.max(vals), vals[0]

    def search_my_move(self, env: GamePlay, is_root_node=False) -> float:

        if env.game_is_over():
            if env.state.player() == 0:
                if env.state.winner == PIECE_WHITE: #white wins
                    return 1
                elif env.state.winner == PIECE_BLACK:  # black wins
                    return -1
            else:
                if env.state.winner == PIECE_WHITE:  # white wins
                    return -1
                elif env.state.winner == PIECE_BLACK:  # black wins
                    return 1
            return 5
        elif env.state.turn >= MAX_GAME_LENGTH:
            # print("EMD")
            return 5

        state = env.state_key

        with self.node_lock[state]:
            if state not in self.tree:
                if self.none_queue:
                    leaf_p, leaf_v = self.expand_and_evaluate(env)
                    # if is_root_node:
                    #     print(leaf_v)
                else:
                    leaf_p, leaf_v = self.expand_and_evaluate_with_net(env)

                self.tree[state].p = leaf_p
                return leaf_v # I'm returning everything from the POV of side to move

            # SELECT STEP
            action_t = self.select_action_q_and_u(env, is_root_node)

            my_visit_stats = self.tree[state]
            my_stats = my_visit_stats.a[action_t]

            my_visit_stats.sum_n += virtual_loss
            my_stats.n += virtual_loss
            my_stats.w += -virtual_loss
            my_stats.q = my_stats.w / my_stats.n
        # if is_root_node:
        #     print(action_t, env.state.turn)

        # print(env.decode_action(action_t), env.state.turn)
        env.move(action_t)

        if env.state.turn > self.max_depth:
            self.max_depth = env.state.turn
        leaf_v = self.search_my_move(env)  # next move from enemy POV

        reach_max = False
        if leaf_v == 5:
            leaf_v = 1
            reach_max = True
        leaf_v = -leaf_v

        # BACKUP STEP
        # on returning search path
        # update: N, W, Q
        # print(state)

        with self.node_lock[state]:
            my_visit_stats.sum_n += -virtual_loss + 1
            my_stats.n += -virtual_loss + 1
            my_stats.w += virtual_loss + leaf_v
            my_stats.q = my_stats.w / my_stats.n

            # if is_root_node:
            #     if env.state.player() == 1 and leaf_v <= 0:
            #         print(my_stats.w, leaf_v)
            #     if env.state.player() == 0 and leaf_v > 0:
            #         print(my_stats.w, leaf_v)
            # if is_root_node:
            # print(f"{player} A: {action_t} ===== ", my_stats.w, leaf_v, env.state.turn)

        if reach_max:
            leaf_v = 5

        return leaf_v

    def expand_and_evaluate_with_net(self, env) -> (np.ndarray, float):
        board_state = env.encode_board()
        board_state = board_state.transpose(2, 0, 1)
        board_state = torch.from_numpy(board_state).float().cuda()
        board_state = torch.unsqueeze(board_state, 0)
        leaf_p, leaf_v = self.net(board_state)
        leaf_p = leaf_p.detach().cpu().numpy().reshape(-1)
        leaf_v = leaf_v.detach().cpu().numpy().reshape(-1)
        # print(leaf_v, env.state.turn)
        return leaf_p, leaf_v

    def expand_and_evaluate(self, env) -> (np.ndarray, float):
        """ expand new leaf, this is called only once per state
        this is called with state locked
        insert P(a|s), return leaf_v
        This gets a prediction for the policy and value of the state within the given env
        :return (float, float): the policy and value predictions for this state
        """
        r_e = 0.75
        board_state = env.encode_board()
        # total_white = np.sum(board_state[:, :, 32])
        # total_black = np.sum(board_state[:, :, 33])
        #
        # reward = np.clip((total_black - total_white)/4, -1, 1)
        leaf_p, leaf_v = self.predict(board_state)

        # if self.reward:
        #     leaf_v = (1 - r_e) * reward + r_e * leaf_v

        return leaf_p, leaf_v

    def predict(self, board_state):
        """
        Gets a prediction from the policy and value network
        :param state_planes: the observation state represented as planes
        :return (float,float): policy (prior probability of taking the action leading to this state)
            and value network (value of the state) prediction for this state.
        """
        pipe = self.pipe_pool.pop()
        pipe.send(board_state)
        ret = pipe.recv()
        self.pipe_pool.append(pipe)
        return ret

    #@profile
    def select_action_q_and_u(self, env, is_root_node):

        # this method is called with state locked
        state = env.state_key
        actions = env.actions()
        if len(actions) == 0:
            return -1

        my_visitstats = self.tree[state]

        if my_visitstats.p is not None: #push p to edges
            tot_p = 1e-8
            for mov in actions:

                mov_p = my_visitstats.p[mov]
                my_visitstats.a[mov].p = mov_p
                tot_p += mov_p
            for a_s in my_visitstats.a.values():
                a_s.p /= tot_p
            my_visitstats.p = None

        xx_ = np.sqrt(my_visitstats.sum_n + 1)  # sqrt of sum(N(s, b); for all b)

        e = noise_eps

        dir_alpha = dirichlet_alpha
        best_s = -999
        best_a = None
        if is_root_node:
            noise = np.random.dirichlet([dir_alpha] * len(my_visitstats.a))
        # np.random.dirichlet([0.1] * 10)
        i = 0
        for action, a_s in my_visitstats.a.items():
            p_ = a_s.p
            if is_root_node:
                p_ = (1-e) * p_ + e * noise[i]
                i += 1
            b = a_s.q + c_puct * p_ * xx_ / (1 + a_s.n)
            if b > best_s:
                best_s = b
                best_a = action
        return best_a

    def apply_temperature(self, policy, turn):
        tau = np.power(tau_decay_rate, turn)
        if tau < 0.1:
            tau = 0
        if tau == 0:
            action = np.argmax(policy)
            ret = np.zeros(ACTION_SPACE)
            ret[action] = 1.0
            return ret
        else:
            ret = np.power(policy, 1/tau)
            ret /= np.sum(ret)
            return ret

    def calc_policy(self, env):
        """calc π(a|s0)
        :return list(float): a list of probabilities of taking each action, calculated based on visit counts.
        """
        state = env.state_key
        my_visitstats = self.tree[state]
        policy = np.zeros(ACTION_SPACE)
        policy_t = np.zeros(ACTION_SPACE)
        w = []
        for action, a_s in my_visitstats.a.items():
            policy[action] = a_s.n
            policy_t[action] = a_s.p
            w.append(a_s.w)

        # if np.max(w) < 0:
        #     print(np.max(w))
        #     for action, a_s in my_visitstats.a.items():
        #         print(a_s.n)

        sum_all = np.sum(policy)
        policy /= np.sum(policy)
        if np.max(w) < 0:
            policy = policy_t
        return policy, sum_all

    def finish_game(self, z):
        """
        When game is done, updates the value of all past moves based on the result.
        :param self:
        :param z: win=1, lose=-1, draw=0
        :return:
        """
        for move in self.moves:  # add this game winner result to all past moves.
            move += [z]


#
#
# net = ChessNet()
# cuda = torch.cuda.is_available()
# if cuda:
#     net.cuda()
# net.share_memory()
# net.eval()
#
# save_as = "v2_4.pth.tar"
# current_net_filename = os.path.join("./model_data/", \
#                                     save_as)
# checkpoint = torch.load(current_net_filename)
# net.load_state_dict(checkpoint['state_dict'])
#
#
#
# board = GamePlay(HEIGHT_MAP=HEIGHT - 100, WIDTH_MAP=WIDTH - 500)
# players = ChessPlayer(net)
# p = players.action(board)
# # board.actions()
# # board.move(597)
# #
# my_visitstats = players.tree[board.state_key]
# # my_visitstats.sum_n
# # policy = np.zeros(ACTION_SPACE)
# #
# # for action, a_s in my_visitstats.a.items():
# #     # policy[white.move_lookup[action]] = a_s.n
# #     print(policy[action])
# #     policy[action]= a_s.n
# # policy /= np.sum(policy)
#
# p = players.calc_policy(board)
# for action, a_s in my_visitstats.a.items():
#     print(a_s.q)
#
#     # policy[white.move_lookup[action]] = a_s.n
#     print(p[action])
#
# m = Manager()
#
# cur_pipes = m.list(
#     [self.current_model.get_pipes(self.config.play.search_threads) for _ in range(self.config.play.max_processes)])

