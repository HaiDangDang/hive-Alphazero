"""
Everything related to configuration of running this application
"""

import os
import string
import numpy as np
MAX_MAP_HAFT = 6
MAX_MAP_FULL = MAX_MAP_HAFT * 2
ACTION_SPACE = MAX_MAP_FULL * MAX_MAP_FULL * 11

number = np.arange(1, 27, 1)
index_number = list(map(str, number))
index_char = list(string.ascii_uppercase)

index_number = index_number[12-MAX_MAP_HAFT:12+MAX_MAP_HAFT]
index_char = index_char[13-MAX_MAP_HAFT:13+MAX_MAP_HAFT]
# index_number[MAX_MAP_HAFT]
# index_char[MAX_MAP_HAFT]
# index_number[MAX_MAP_HAFT]
STATE_FEATURES = 56

MAX_GAME_LENGTH = 55
MAX_LEN_BACK = 5

SEARCH_THREADS = 32

MAX_PROCESS = 60

BOT_WEIGHT = 0.24

LOSS_WEIGHT = {'value': 1.0,
               'policy': 1.0}

DISCOUNTED_REWARD = 0.99