from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from threading import Lock
import numpy as np

from hive_engine.config import Config
from hive_engine.env_hive import GamePlay
import torch

class VisitStats:

    def __init__(self):
        self.a = defaultdict(ActionStats)
        self.sum_n = 0


class ActionStats:

    def __init__(self):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0

class HivePlayer:

    def __init__(self,  config: Config, pipes=None):
        self.env = GamePlay()
        self.config = config

        self.play_config =  self.config.play
        self.pipe_pool = pipes

        self.tree = defaultdict(VisitStats)

        self.node_lock = defaultdict(Lock)

    def action(self, can_stop=True):
        root_value, naked_value = self.search_moves(env)

    def search_moves(self, env) -> (float, float):
        futures = []
        with ThreadPoolExecutor(max_workers=self.play_config.search_threads) as executor:
            for _ in range(self.play_config.simulation_num_per_move):
                futures.append(executor.submit(self.search_my_move,env=env.copy(),is_root_node=True))

    def search_my_move(self, env: ChessEnv, is_root_node=False) -> float:

        if env.game_is_over():
            if env.winner == Winner.draw:
                return 0
            # assert env.whitewon != env.white_to_move # side to move can't be winner!
            return -1

        state = env.state_to_str()

        with self.node_lock[state]:
            if state not in self.tree:
                leaf_p, leaf_v = self.expand_and_evaluate(env)
                self.tree[state].p = leaf_p
                return leaf_v  # I'm returning everything from the POV of side to move

            # SELECT STEP
            action_t = self.select_action_q_and_u(env, is_root_node)

            virtual_loss = self.play_config.virtual_loss

            my_visit_stats = self.tree[state]
            my_stats = my_visit_stats.a[action_t]

            my_visit_stats.sum_n += virtual_loss
            my_stats.n += virtual_loss
            my_stats.w += -virtual_loss
            my_stats.q = my_stats.w / my_stats.n

        env.step(action_t.uci())
        leaf_v = self.search_my_move(env)  # next move from enemy POV
        leaf_v = -leaf_v

        # BACKUP STEP
        # on returning search path
        # update: N, W, Q
        with self.node_lock[state]:
            my_visit_stats.sum_n += -virtual_loss + 1
            my_stats.n += -virtual_loss + 1
            my_stats.w += virtual_loss + leaf_v
            my_stats.q = my_stats.w / my_stats.n

        return leaf_v

    def expand_and_evaluate(self, env) -> (np.ndarray, float):

        state_planes = env.encode_board()
        state_planes = state_planes.transpose(2,0,1)
        state_planes = torch.from_numpy(state_planes).float().cuda()

        leaf_p, leaf_v = self.predict(state_planes)
        # these are canonical policy and value (i.e. side to move is "white")

        if not env.white_to_move:
            leaf_p = Config.flip_policy(leaf_p)  # get it back to python-chess form

        return leaf_p, leaf_v

    def predict(self, state_planes):

        pipe = self.pipe_pool.pop()
        pipe.send(state_planes)
        ret = pipe.recv()
        self.pipe_pool.append(pipe)
        return ret

    def select_action_q_and_u(self, env, is_root_node) -> chess.Move:

        # this method is called with state locked
        state = env.state_to_str()

        my_visitstats = self.tree[state]

        if my_visitstats.p is not None: #push p to edges
            tot_p = 1e-8
            for mov in env.board.legal_moves:
                mov_p = my_visitstats.p[self.move_lookup[mov]]
                my_visitstats.a[mov].p = mov_p
                tot_p += mov_p
            for a_s in my_visitstats.a.values():
                a_s.p /= tot_p
            my_visitstats.p = None

        xx_ = np.sqrt(my_visitstats.sum_n + 1)  # sqrt of sum(N(s, b); for all b)

        e = self.play_config.noise_eps
        c_puct = self.play_config.c_puct
        dir_alpha = self.play_config.dirichlet_alpha

        best_s = -999
        best_a = None
        if is_root_node:
            noise = np.random.dirichlet([dir_alpha] * len(my_visitstats.a))

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