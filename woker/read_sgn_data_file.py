import os, zipfile
import textract
import pickle
extension = ".txt"
dir_name = '../ABC/hivegames'

extension = ".zip"
html_extension = ".html"
database = '../database'
datahist = '../datahist'

#
# for idx, file in enumerate(os.listdir(dir_name)):
#     filename = os.path.join(dir_name, file)
#
#     for item in os.listdir():  # loop through items in dir
#
#         if item.endswith(extension):  # check for ".zip" extension
#             os.remove(item)
#             text = None
#             try:
#                 text = textract.process(item, encoding='ISO-8859-1')
#             except:
#                 a = 0
#             if test is not None:
#                 print(text)
#             with open(item) as f:
#
#                 collection = sgf.parse(f.read())
#
#             print(filename)
#             break
#             # thisFile = "HV-coccino-Dumbot-2022-04-19-1233.sgf"
#             base = os.path.splitext(filename)[0]
#             os.rename(filename, base + ".txt")
#
#             with open(filename) as f:
#                 collection = sgf.parse(f.read())
#                 break
#
#             with open(filename, "rb") as f:
#                 game = sgf.Sgf_game.from_bytes(f.read())
#
#
#     for item in os.listdir(filename):  # loop through items in dir
#
#
# for idx, file in enumerate(os.listdir(dir_name)):
#     filename = os.path.join(dir_name, file)
#     if not filename.endswith(html_extension):
#         for item in os.listdir(filename):  # loop through items in dir
#             if item.endswith(extension):  # check for ".zip" extension
#                 x = os.path.join(filename, item)
#                 file_name = os.path.abspath(x)  # get full path of files
#                 zip_ref = zipfile.ZipFile(file_name)  # create zipfile object
#                 zip_ref.extractall(database)  # extract file to dir
#                 zip_ref.close()  # close file
#             if item.endswith(html_extension):  # check for ".zip" extension
#                 f_x = os.path.join(filename, item)
#                 s_x = os.path.join(database, item)
#                 os.rename(f_x, f"{s_x[:-8]}sgf")
#     if filename.endswith(html_extension):
#         s_x = os.path.join(database, file)
#         os.rename(filename, f"{s_x[:-8]}sgf")
#
#
#
# extension = ".sgf"
# for idx, file in enumerate(os.listdir(database)):
#     filename = os.path.join(database, file)
#     if filename.endswith(extension):  # check for ".zip" extension
#         os.rename(filename, f"{filename[:-3]}txt")


extension = ".txt"
count = 0

import numpy as np
import string

index_char = list(string.ascii_uppercase)
number = np.arange(1, 27, 1)
index_number = list(map(str, number))
center_y = {}


stack ={}
tile_key = {}

key = ('N','14')
tile_key[key] = key
stack[key] = [key]
count = 2
index_x = index_char.index(key[0])
index_y = index_number.index(key[1])
while True:
    for i in range(count):
        x = index_x - i
        y = index_y + count - i - 1
        add_ = tuple((index_char[x], index_number[y]))
        tile_key[add_] = key
        print(add_)
        stack[key].append(add_)
    print("----")
    count += 1
    if count > 11:
        break

key = ('O','14')
stack[key] = [key]
tile_key[key] = key

count = 2
index_x = index_char.index('O')
index_y = index_number.index('14')

while True:
    for i in range(count):
        x = index_x - i + count - 1
        y = index_y  + count - 1
        add_ = tuple((index_char[x], index_number[y]))
        tile_key[add_] = key
        print(add_)
        stack[('O','14') ].append(add_)
    print("----")
    count+= 1
    if count>11:
        break

key = ('O','13')
stack[key] = [key]
tile_key[key] = key

count = 2
index_x = index_char.index('O')
index_y = index_number.index('13')

while True:
    for i in range(count):
        x = index_x + count - 1
        y = index_y  + i
        add_ = tuple((index_char[x], index_number[y]))
        tile_key[add_] = key

        print(add_)
        stack[('O','13') ].append(add_)
    print("----")
    count+= 1
    if count>11:
        break

key = ('N','12')
stack[key] = [key]
tile_key[key] = key

count = 2
index_x = index_char.index('N')
index_y = index_number.index('12')

while True:
    for i in range(count):
        x = index_x + i
        y = index_y - count + 1 + i
        add_ = tuple((index_char[x], index_number[y]))
        tile_key[add_] = key

        print(add_)
        stack[('N','12') ].append(add_)
    print("----")
    count+= 1
    if count>11:
        break

key = ('M','12')
stack[key] = [key]
tile_key[key] = key

count = 2
index_x = index_char.index('M')
index_y = index_number.index('12')

while True:
    for i in range(count):
        x = index_x - count + 1 + i
        y = index_y - count + 1
        add_ = tuple((index_char[x], index_number[y]))
        tile_key[add_] = key

        print(add_)
        stack[('M','12') ].append(add_)
    print("----")
    count+= 1
    if count>11:
        break

key = ('M','13')
stack[key] = [key]
tile_key[key] = key

count = 2
index_x = index_char.index('M')
index_y = index_number.index('13')

while True:
    for i in range(count):
        x = index_x - count + 1
        y = index_y   - i
        add_ = tuple((index_char[x], index_number[y]))
        tile_key[add_] = key

        print(add_)
        stack[('M','13')].append(add_)
    print("----")
    count+= 1
    if count>11:
        break

for i in index_char:
    center_y[i] = []
piece_types = ["G1", "G2", "G3", "A1", "A2", "A3", "S1", "S2", "B1", "B2", "Q", ]

piece_types_2 = ["G1", "G2", "G3", "A1", "A2", "A3", "S1", "S2", "B1", "B2", "Q",
                 "g1", "g2", "g3", "a1", "a2", "a3", "s1", "s2", "b1", "b2", "q",]
center_2 = ('M','13')
stack_key = list(stack.keys())
def find_type(line):
    new_lines = line
    new_index = 0
    c_index = 999
    index_piece = ""
    for piece in piece_types_2:
        index = new_lines.find(piece)
        if index != -1:
            if index < c_index:
                c_index = index
                # print(piece)
                index_piece= piece
                for j in range(10):
                    if index + j > len(new_lines) -1:
                        break
                    if new_line[index + j] == " ":
                        new_index = index + j
                        break
    # print( line[new_index+1:])
    chars = line[new_index+1:].split(" ")
    x = chars[0]
    y = chars[1]
    return index_piece.upper(),x,y
from copy import deepcopy
count = 0
LENS = []
for idx, file in enumerate(os.listdir(database)):
    # if idx < 187368:
    #     continue
    filename = os.path.join(database, file)
    file_hist = os.path.join(datahist, file)

    if filename.endswith(extension):  # check for ".zip" extension
        file = open(filename, 'r', encoding="ISO-8859-1")
        contents = [line for line in file.readlines()]


        expanse = False
        contents_join = " ".join(contents)
        expanse_piece_L = ['L1', 'wL1', 'WL1']
        find_ladyBug = False
        find_M = False
        for ex_piece in expanse_piece_L:
            index_ex = contents_join.find(ex_piece)
            if index_ex != -1:
                find_ladyBug = True
                break

        expanse_piece_M = ['M1', 'wM1', 'WM1', 'bM1', 'BM1', 'bP1', 'wP1', 'BP1', 'WP1',
                           'ropb p', 've B P', 'wp', 'Wp', 'Bp', 'bp', 'wS3','W P','w P','b P','B P',
                           'wG4', 'w?1', 'wA4', 'wL2']

        for ex_piece in expanse_piece_M:
            index_ex = contents_join.find(ex_piece)
            if index_ex != -1:
                find_M = True
                break

        expanse = find_ladyBug or find_M


        center_x = ""
        center_n = ""
        check = False

        if not expanse:
            boards_full = None

            boards_W = {}
            boards_B = {}
            for p in piece_types:
                boards_B[p] = None
                boards_W[p] = None
            players = {'W': boards_W, 'B': boards_B}
            boards = np.zeros((1, 26, 27), dtype=np.dtype('<U16'))

            histories = []
            shift_x = 0
            shift_y = 0
            c_l = 0
            x_list = []
            y_list = []
            P0 = 0
            P1 = 0
            rotation_x = 0
            rotation_y = 0
            skip_this = False
            for line in contents:
                index_id = line.find("P0[id ")
                if index_id != -1:
                    index_bot = line.find("bot")
                    if index_bot != -1:
                        P0 = 1
                index_id = line.find("P1[id ")
                if index_id != -1:
                    index_bot = line.find("bot")
                    if index_bot != -1:
                        P1 = 1

                if line[:3] == '; P':
                    index_drop = line.find("dropb")
                    index_move = line.find("move")
                    index_move_done = line.find("movedone")

                    player = 'W'
                    if line.find("P0") == -1:
                        player = 'B'
                    index_done = contents[c_l+1].find("done")
                    if ((index_drop != -1 and index_done != -1 )
                        or (index_move != -1 and index_done != -1)
                        or index_move_done != -1):
                        if index_drop != -1:
                            start = index_drop +6
                        else:
                            start = index_move + 5
                        new_line = line[start:]
                        piece, x, y = find_type(new_line)

                        if len(histories) == 0:
                            shift_x = index_char.index(x) - index_char.index("N")
                            shift_y = index_number.index(y) - index_number.index("13")

                            # if shift_x != 0:
                            #     print(idx, x)


                        # print(piece,x,y)
                        center_y[x].append(int(y))

                        x = (index_char.index(x) - shift_x) % 26
                        y = (index_number.index(y) - shift_y) % 26

                        delta_x = x - index_char.index("N")
                        delta_y = y - index_number.index("13")
                        if delta_x >= 8 or delta_x <= -8 or \
                                delta_y >= 8 or delta_y <= -8:
                            skip_this = True
                            break

                        if len(histories) >= 1:
                            x = index_char[x]
                            y = index_number[y]
                            tuple_x_y = tuple((x, y))
                            if len(histories) == 1:
                                # print(tuple_x_y)
                                rotation = stack_key.index(center_2) - stack_key.index(tuple_x_y)
                            if (x,y) != ('N','13'):
                                location_x_y = tile_key[(x,y)]
                                location_x_y_index = stack[location_x_y].index((x,y))
                                location_x_y = stack_key[(stack_key.index(location_x_y) + rotation)%6]
                                new_x_y = stack[location_x_y][location_x_y_index]
                                x = new_x_y[0]
                                y = new_x_y[1]
                            x = index_char.index(x)
                            y = index_number.index(y)


                        if not (0 <= x <= 25):
                            print(x)
                        if not (0 <= y <= 25):
                            print(y)

                        x_list.append(x - index_char.index("N"))
                        y_list.append(y - index_number.index("13"))

                        x = index_char[x]
                        y = index_number[y]

                        if player == "W":
                            histories.append([piece, x, y, player, P0])
                            last = 'W'
                        else:
                            histories.append([piece, x, y, player, P1])
                            last = 'B'

                c_l += 1

            # print(P0, P1)
            if len(histories) >= 6:
                if np.max(x_list) <= 5 and np.min(x_list) >= -6 and\
                    np.max(y_list) <= 5 and np.min(y_list) >= -6 and not skip_this and P0 == 1:
                    count += len(histories)
                    LENS.append(len(histories))
                    # file_hist = file_hist[:-4]
                    # with open(file_hist + '.pkl', 'wb') as f:
                    #     pickle.dump(histories, f)
            # if boards_full is not None:
            #     file_hist = file_hist[:-4]
            #     print(boards_full.shape)
            #     with open(file_hist + '.npy', 'wb') as f:
            #         np.save(f, boards_full)

                        # check = True
                        # break
        #     if check:
        #         break
        #
        # break

len(LENS)
# np.sum(LENS)
# np.percentile(LENS, 91)
# 578580
# for k,v in center_y.items():
#     print(k, np.max(v), np.min(v))
#
# np.mean(LENS)
# np.std(LENS)
# np.percentile(LENS,99.9)
# 41245

#
# rotation = stack_key.index(center_2) - stack_key.index(('O','14'))
#
# x = 'R'
# y = '9'
# location_x_y = tile_key[(x, y)]
# location_x_y_index = stack[location_x_y].index((x, y))
# location_x_y = stack_key[(stack_key.index(location_x_y) + rotation)%6]
# new_x_y = stack[location_x_y][location_x_y_index]
# print(new_x_y)