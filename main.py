#!/usr/bin/env pypy
# -*- coding: utf-8 -*-

from __future__ import print_function
import re, sys, time
from itertools import count
from collections import namedtuple
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#############################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras import datasets
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
threshold=0.7
import time
###################################################################################
#load model
model_architecture = "chess.json"
model_weights = "chess_weight.h5"
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)
###################################################################################
# CAC CHUONG TRINH CON CUA CNN

def preProcessing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255
    return img
def chessname(chiso):
    i=chiso
    name="noname"
    if i==0 :
        name = "XE_DEN"
    if i==1 :
        name = "XE_DO"
    if i==2 :
        name = "MA_DEN"
    if i==3 :
        name = "MA_DO"
    if i==4 :
        name=  "TINH_DEN"
    if i==5 :
        name = "TINH_DO"
    if i==6 :
        name = "SI_DEN"
    if i==7 :
        name = "SI_DO"
    if i==8 :
        name = "TUONG_DEN"
    if i==9 :
        name = "TUONG_DO"
    if i==10 :
        name = "PHAO_DEN"
    if i==11 :
        name = "PHAO_DO"
    if i==12 :
        name = "TOT_DEN"
    if i==13 :
        name = "TOT_DO"
    return name

def proset_img(img_0):
    img = img_0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(gray, (9, 9),2)  # ADD GAUSSIAN BLUR
    imgThreshold = cv2.Canny(imgBlur, 0, 25)
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    areamax = 0
    xct = 50
    yct = 50
    wct = 10
    hct = 10
    for cnt in contours:
        area = cv2.contourArea(cnt)
       # print(area)
        if area > 50:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            if objCor > 4:
                if area > areamax:
                    areamax = area
                    xct = x
                    yct = y
                    wct = w
                    hct = h
    imgROI = img[yct:yct + hct , xct :xct + wct ]
    return imgROI

def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)  # ADD GAUSSIAN BLUR
   # imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 5)  # APPLY ADAPTIVE THRESHOLD
    imgThreshold = cv2.Canny(imgBlur, 0, 100)
    return imgThreshold


#### 3 - Reorder points for Warp Perspective
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


#### 3 - FINDING THE BIGGEST COUNTOUR ASSUING THAT IS THE SUDUKO PUZZLE
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
       # print(area)
        if area > 50000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
  #  print("ma")
    return biggest,max_area


#### 4 - TO SPLIT THE IMAGE INTO 81 DIFFRENT IMAGES
def splitBoxes(img):
    rows = np.vsplit(img,10)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes

def kiemtradauO(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(gray, (9, 9), 2)  # ADD GAUSSIAN BLUR
    imgThreshold = cv2.Canny(imgBlur, 100, 200)
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    tron=0
    for cnt in contours:
        area = cv2.contourArea(cnt)
      #  print(area)
        if area > 100:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            objCor = len(approx)
            if objCor > 5:
                tron=1
    if tron ==1:
        return 1
    else:
        return 0
def nameposition(vitri):
    chiso=vitri
    if chiso<=8:
        chiso=chiso+1
    elif chiso<=17:
        chiso=chiso+2
    elif chiso<=26:
        chiso=chiso+3
    elif chiso<=35:
        chiso=chiso+4
    elif chiso<=44:
        chiso=chiso+5
    elif chiso<=53:
        chiso=chiso+6
    elif chiso<=62:
        chiso=chiso+7
    elif chiso<=71:
        chiso=chiso+8
    elif chiso<=80:
        chiso=chiso+9
    else :
        chiso=chiso+10
    a=chiso%10
    b=chiso//10
    toadoX=a*240-120
    toadoY=b*240+120
    return (toadoX,toadoY)




#################################################################################
# CAC CHUONG TRINH CON CUA MINIMAX
piece = { 'P': 44, 'N': 108, 'B': 23, 'R': 233, 'A': 23, 'C': 101, 'K': 2500}

# 子力价值表参考“象眼”

pst = {
    "P": (
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  9,  9,  9, 11, 13, 11,  9,  9,  9,  0,  0,  0,  0,
      0,  0,  0, 19, 24, 34, 42, 44, 42, 34, 24, 19,  0,  0,  0,  0,
      0,  0,  0, 19, 24, 32, 37, 37, 37, 32, 24, 19,  0,  0,  0,  0,
      0,  0,  0, 19, 23, 27, 29, 30, 29, 27, 23, 19,  0,  0,  0,  0,
      0,  0,  0, 14, 18, 20, 27, 29, 27, 20, 18, 14,  0,  0,  0,  0,
      0,  0,  0,  7,  0, 13,  0, 16,  0, 13,  0,  7,  0,  0,  0,  0,
      0,  0,  0,  7,  0,  7,  0, 15,  0,  7,  0,  7,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0, 11, 15, 11,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
    ),
    "B":(
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0, 40,  0,  0,  0, 40,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0, 38,  0,  0, 40, 43, 40,  0,  0, 38,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0, 43,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0, 40, 40,  0, 40, 40,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
    ),
    "N": (
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0, 90, 90, 90, 96, 90, 96, 90, 90, 90,  0,  0,  0,  0,
      0,  0,  0, 90, 96,103, 97, 94, 97,103, 96, 90,  0,  0,  0,  0,
      0,  0,  0, 92, 98, 99,103, 99,103, 99, 98, 92,  0,  0,  0,  0,
      0,  0,  0, 93,108,100,107,100,107,100,108, 93,  0,  0,  0,  0,
      0,  0,  0, 90,100, 99,103,104,103, 99,100, 90,  0,  0,  0,  0,
      0,  0,  0, 90, 98,101,102,103,102,101, 98, 90,  0,  0,  0,  0,
      0,  0,  0, 92, 94, 98, 95, 98, 95, 98, 94, 92,  0,  0,  0,  0,
      0,  0,  0, 93, 92, 94, 95, 92, 95, 94, 92, 93,  0,  0,  0,  0,
      0,  0,  0, 85, 90, 92, 93, 78, 93, 92, 90, 85,  0,  0,  0,  0,
      0,  0,  0, 88, 85, 90, 88, 90, 88, 90, 85, 88,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
    ),
    "R": (
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,206,208,207,213,214,213,207,208,206,  0,  0,  0,  0,
      0,  0,  0,206,212,209,216,233,216,209,212,206,  0,  0,  0,  0,
      0,  0,  0,206,208,207,214,216,214,207,208,206,  0,  0,  0,  0,
      0,  0,  0,206,213,213,216,216,216,213,213,206,  0,  0,  0,  0,
      0,  0,  0,208,211,211,214,215,214,211,211,208,  0,  0,  0,  0,
      0,  0,  0,208,212,212,214,215,214,212,212,208,  0,  0,  0,  0,
      0,  0,  0,204,209,204,212,214,212,204,209,204,  0,  0,  0,  0,
      0,  0,  0,198,208,204,212,212,212,204,208,198,  0,  0,  0,  0,
      0,  0,  0,200,208,206,212,200,212,206,208,200,  0,  0,  0,  0,
      0,  0,  0,194,206,204,212,200,212,204,206,194,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
    ),
    "C": (
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,100,100, 96, 91, 90, 91, 96,100,100,  0,  0,  0,  0,
      0,  0,  0, 98, 98, 96, 92, 89, 92, 96, 98, 98,  0,  0,  0,  0,
      0,  0,  0, 97, 97, 96, 91, 92, 91, 96, 97, 97,  0,  0,  0,  0,
      0,  0,  0, 96, 99, 99, 98,100, 98, 99, 99, 96,  0,  0,  0,  0,
      0,  0,  0, 96, 96, 96, 96,100, 96, 96, 96, 96,  0,  0,  0,  0,
      0,  0,  0, 95, 96, 99, 96,100, 96, 99, 96, 95,  0,  0,  0,  0,
      0,  0,  0, 96, 96, 96, 96, 96, 96, 96, 96, 96,  0,  0,  0,  0,
      0,  0,  0, 97, 96,100, 99,101, 99,100, 96, 97,  0,  0,  0,  0,
      0,  0,  0, 96, 97, 98, 98, 98, 98, 98, 97, 96,  0,  0,  0,  0,
      0,  0,  0, 96, 96, 97, 99, 99, 99, 97, 96, 96,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
    )
}

pst["A"] = pst["B"]
pst["K"] = pst["P"]
pst["K"] = [i + piece["K"] if i > 0 else 0 for i in pst["K"]]

A0, I0, A9, I9 = 12 * 16 + 3,12 * 16 + 11, 3 * 16 + 3,  3 * 16 + 11



# Lists of possible moves for each piece type.
N, E, S, W = -16, 1, 16, -1
directions = {
    'P': (N, W, E),
    'N': (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    'B': (2 * N + 2 * E, 2 * S + 2 * E, 2 * S + 2 * W, 2 * N + 2 * W),
    'R': (N, E, S, W),
    'C': (N, E, S, W),
    'A': (N+E, S+E, S+W, N+W),
    'K': (N, E, S, W)
}

MATE_LOWER = piece['K'] - (2*piece['R'] + 2*piece['N'] + 2*piece['B'] + 2*piece['A'] + 2*piece['C'] + 5*piece['P'])
MATE_UPPER = piece['K'] + (2*piece['R'] + 2*piece['N'] + 2*piece['B'] + 2*piece['A'] + 2*piece['C'] + 5*piece['P'])

# The table size is the maximum number of elements in the transposition table.
TABLE_SIZE = 1e7

# Constants for tuning search
QS_LIMIT = 219
EVAL_ROUGHNESS = 13
DRAW_TEST = True
THINK_TIME = 5

###############################################################################
# Chess logic
###############################################################################

class Position(namedtuple('Position', 'board score')):
    """ A state of a chess game
    board -- a 256 char representation of the board
    score -- the board evaluation
    """
    def gen_moves(self):
        # For each of our pieces, iterate through each possible 'ray' of moves,
        # as defined in the 'directions' map. The rays are broken e.g. by
        # captures or immediately in case of pieces such as knights.
        for i, p in enumerate(self.board):
            if p == 'K':
                for scanpos in range(i - 16,A9,-16):
                    if self.board[scanpos] == 'k':
                        yield (i,scanpos)
                    elif self.board[scanpos] != '.':
                        break
            if not p.isupper(): continue
            if p == 'C':
                for d in directions[p]:
                    cfoot = 0
                    for j in count(i+d, d):
                        q = self.board[j]
                        if q.isspace():break
                        if cfoot == 0 and q == '.':yield (i,j)
                        elif cfoot == 0 and q != '.':cfoot += 1
                        elif cfoot == 1 and q.islower(): yield (i,j);break
                        elif cfoot == 1 and q.isupper(): break;
                continue
            for d in directions[p]:
                for j in count(i+d, d):
                    q = self.board[j]
                    # Stay inside the board, and off friendly pieces
                    if q.isspace() or q.isupper(): break
                    # 过河的卒/兵才能横着走
                    if p == 'P' and d in (E, W) and i > 128: break
                    # j & 15 等价于 j % 16但是更快
                    elif p in ('A','K') and (j < 160 or j & 15 > 8 or j & 15 < 6): break
                    elif p == 'B' and j < 128: break
                    elif p == 'N':
                        n_diff_x = (j - i) & 15
                        if n_diff_x == 14 or n_diff_x == 2:
                            if self.board[i + (1 if n_diff_x == 2 else -1)] != '.': break
                        else:
                            if j > i and self.board[i + 16] != '.': break
                            elif j < i and self.board[i - 16] != '.': break
                    elif p == 'B' and self.board[i + d // 2] != '.':break
                    # Move it
                    yield (i, j)
                    # Stop crawlers from sliding, and sliding after captures
                    if p in 'PNBAK' or q.islower(): break

    def rotate(self):
        ''' Rotates the board, preserving enpassant '''
        return Position(
            self.board[-2::-1].swapcase() + " ", -self.score)

    def nullmove(self):
        ''' Like rotate, but clears ep and kp '''
        return self.rotate()

    def move(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        put = lambda board, i, p: board[:i] + p + board[i+1:]
        # Copy variables and reset ep and kp
        board = self.board
        score = self.score + self.value(move)
        # Actual move
        board = put(board, j, board[i])
        board = put(board, i, '.')
        return Position(board, score).rotate()

    def value(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        # Actual move
        score = pst[p][j] - pst[p][i]
        # Capture
        if q.islower():
            score += pst[q.upper()][255-j-1]
        return score

###############################################################################
# Search logic
###############################################################################

# lower <= s(pos) <= upper
Entry = namedtuple('Entry', 'lower upper')

class Searcher:
    def __init__(self):
        self.tp_score = {}
        self.tp_move = {}
        self.history = set()
        self.nodes = 0

    def bound(self, pos, gamma, depth, root=True):
        """ returns r where
                s(pos) <= r < gamma    if gamma > s(pos)
                gamma <= r <= s(pos)   if gamma <= s(pos)"""
        self.nodes += 1

        # Depth <= 0 is QSearch. Here any position is searched as deeply as is needed for
        # calmness, and from this point on there is no difference in behaviour depending on
        # depth, so so there is no reason to keep different depths in the transposition table.
        depth = max(depth, 0)

        # Sunfish is a king-capture engine, so we should always check if we
        # still have a king. Notice since this is the only termination check,
        # the remaining code has to be comfortable with being mated, stalemated
        # or able to capture the opponent king.
        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER

        # We detect 3-fold captures by comparing against previously
        # _actually played_ positions.
        # Note that we need to do this before we look in the table, as the
        # position may have been previously reached with a different score.
        # This is what prevents a search instability.
        # FIXME: This is not true, since other positions will be affected by
        # the new values for all the drawn positions.
        if DRAW_TEST:
            if not root and pos in self.history:
                return 0

        # Look in the table if we have already searched this position before.
        # We also need to be sure, that the stored search was over the same
        # nodes as the current search.
        entry = self.tp_score.get((pos, depth, root), Entry(-MATE_UPPER, MATE_UPPER))
        if entry.lower >= gamma and (not root or self.tp_move.get(pos) is not None):
            return entry.lower
        if entry.upper < gamma:
            return entry.upper

        # Here extensions may be added
        # Such as 'if in_check: depth += 1'

        # Generator of moves to search in order.
        # This allows us to define the moves, but only calculate them if needed.
        def moves():
            # First try not moving at all. We only do this if there is at least one major
            # piece left on the board, since otherwise zugzwangs are too dangerous.
            if depth > 0 and not root and any(c in pos.board for c in 'RNC'):
                yield None, -self.bound(pos.nullmove(), 1-gamma, depth-3, root=False)
            # For QSearch we have a different kind of null-move, namely we can just stop
            # and not capture anythign else.
            if depth == 0:
                yield None, pos.score
            # Then killer move. We search it twice, but the tp will fix things for us.
            # Note, we don't have to check for legality, since we've already done it
            # before. Also note that in QS the killer must be a capture, otherwise we
            # will be non deterministic.
            killer = self.tp_move.get(pos)
            if killer and (depth > 0 or pos.value(killer) >= QS_LIMIT):
                yield killer, -self.bound(pos.move(killer), 1-gamma, depth-1, root=False)
            # Then all the other moves
            for move in sorted(pos.gen_moves(), key=pos.value, reverse=True):
            #for val, move in sorted(((pos.value(move), move) for move in pos.gen_moves()), reverse=True):
                # If depth == 0 we only try moves with high intrinsic score (captures and
                # promotions). Otherwise we do all moves.
                if depth > 0 or pos.value(move) >= QS_LIMIT:
                    yield move, -self.bound(pos.move(move), 1-gamma, depth-1, root=False)

        # Run through the moves, shortcutting when possible
        best = -MATE_UPPER
        for move, score in moves():
            best = max(best, score)
            if best >= gamma:
                # Clear before setting, so we always have a value
                if len(self.tp_move) > TABLE_SIZE: self.tp_move.clear()
                # Save the move for pv construction and killer heuristic
                self.tp_move[pos] = move
                break

        # Stalemate checking is a bit tricky: Say we failed low, because
        # we can't (legally) move and so the (real) score is -infty.
        # At the next depth we are allowed to just return r, -infty <= r < gamma,
        # which is normally fine.
        # However, what if gamma = -10 and we don't have any legal moves?
        # Then the score is actaully a draw and we should fail high!
        # Thus, if best < gamma and best < 0 we need to double check what we are doing.
        # This doesn't prevent sunfish from making a move that results in stalemate,
        # but only if depth == 1, so that's probably fair enough.
        # (Btw, at depth 1 we can also mate without realizing.)
        if best < gamma and best < 0 and depth > 0:
            is_dead = lambda pos: any(pos.value(m) >= MATE_LOWER for m in pos.gen_moves())
            if all(is_dead(pos.move(m)) for m in pos.gen_moves()):
                in_check = is_dead(pos.nullmove())
                best = -MATE_UPPER if in_check else 0

        # Clear before setting, so we always have a value
        if len(self.tp_score) > TABLE_SIZE: self.tp_score.clear()
        # Table part 2
        if best >= gamma:
            self.tp_score[pos, depth, root] = Entry(best, entry.upper)
        if best < gamma:
            self.tp_score[pos, depth, root] = Entry(entry.lower, best)

        return best

    def search(self, pos, history=()):
        """ Iterative deepening MTD-bi search """
        self.nodes = 0
        if DRAW_TEST:
            self.history = set(history)
            # print('# Clearing table due to new history')
            self.tp_score.clear()

        # In finished games, we could potentially go far enough to cause a recursion
        # limit exception. Hence we bound the ply.
        for depth in range(1, 1000):
            # The inner loop is a binary search on the score of the position.
            # Inv: lower <= score <= upper
            # 'while lower != upper' would work, but play tests show a margin of 20 plays
            # better.
            lower, upper = -MATE_UPPER, MATE_UPPER
            while lower < upper - EVAL_ROUGHNESS:
                gamma = (lower+upper+1)//2
                score = self.bound(pos, gamma, depth)
                if score >= gamma:
                    lower = score
                if score < gamma:
                    upper = score
            # We want to make sure the move to play hasn't been kicked out of the table,
            # So we make another call that must always fail high and thus produce a move.
            self.bound(pos, lower, depth)
            # If the game hasn't finished we can retrieve our move from the
            # transposition table.
            yield depth, self.tp_move.get(pos), self.tp_score.get((pos, depth, True),Entry(-MATE_UPPER, MATE_UPPER)).lower

###############################################################################
# User interface
###############################################################################

# Python 2 compatability
if sys.version_info[0] == 2:
    input = raw_input


def parse(c):
    fil, rank = ord(c[0]) - ord('a'), int(c[1])
    return A0 + fil - 16*rank


def render(i):
    rank, fil = divmod(i - A0, 16)
    return chr(fil + ord('a')) + str(-rank)

def print_pos(pos):
    print()
    uni_pieces = {'r':'车', 'n':'马', 'b':'相', 'a':'仕', 'k':'帅', 'p':'兵', 'c':'炮',
                  'R':'俥', 'N':'傌', 'B':'象', 'A':'士', 'K':'将', 'P':'卒', 'C':'砲', '.':'．'}
    for i, row in enumerate(pos.board.split()):
        print(' ', 9-i, ''.join(uni_pieces.get(p, p) for p in row))
    print('    ａｂｃｄｅｆｇｈｉ\n\n')
def print_pos1(pos):
    print()
    uni_pieces = {'r':'车', 'n':'马', 'b':'相', 'a':'仕', 'k':'帅', 'p':'兵', 'c':'炮',
                  'R':'俥', 'N':'傌', 'B':'象', 'A':'士', 'K':'将', 'P':'卒', 'C':'砲', '.':'．'}
    for i, row in enumerate(pos.board.split()):
        print(' ', 9-i, ''.join(uni_pieces.get(p, p) for p in row))
    print('    ａｂｃｄｅｆｇｈｉ\n\n')
def print_pos2(pos):
    print()
    temp_row=[]
    uni_pieces = {'R': '车', 'N': '马', 'B': '相', 'A': '仕', 'K': '帅', 'P': '兵', 'C': '炮',
                  'r': '俥', 'n': '傌', 'b': '象', 'a': '士', 'k': '将', 'p': '卒', 'c': '砲', '.': '．'}
    for i, row in enumerate(pos.board.split()):
        temp_row.append(row)
    for j in range(0,10):
        row=temp_row[9-j]
        row=reversed(row)
        print(' ', 9-j, ''.join(uni_pieces.get(p, p) for p in row))
    print('    ａｂｃｄｅｆｇｈｉ\n\n')

def chess_hientai(mang):
    dl=mang
    initial = (
        '               \n'  # 0 -  9
        '               \n'  # 10 - 19
        '               \n'  # 10 - 19
        '   (0)(1)(2)(3)(4)(5)(6)(7)(8)   \n'  # 20 - 29
        '   (9)(10)(11)(12)(13)(14)(15)(16)(17)   \n'  # 40 - 49
        '   (18)(19)(20)(21)(22)(23)(24)(25)(26)   \n'  # 40 - 49
        '   (27)(28)(29)(30)(31)(32)(33)(34)(35)   \n'  # 30 - 39
        '   (36)(37)(38)(39)(40)(41)(42)(43)(44)   \n'  # 50 - 59
        '   (45)(46)(47)(48)(49)(50)(51)(52)(53)   \n'  # 70 - 79
        '   (54)(55)(56)(57)(58)(59)(60)(61)(62)   \n'  # 80 - 89
        '   (63)(64)(65)(66)(67)(68)(69)(70)(71)   \n'  # 70 - 79
        '   (72)(73)(74)(75)(76)(77)(78)(79)(80)   \n'  # 70 - 79
        '   (81)(82)(83)(84)(85)(86)(87)(88)(89)   \n'  # 90 - 99
        '               \n'  # 100 -109
        '               \n'  # 100 -109
        '               \n'  # 110 -119
    )
    temp=initial
    for i in range(0,90):
        name=dl[i]
        if name == "XE_DICH":
            dl[i]="R"
        if name == "XE_TA":
            dl[i]= "r"
        if name == "MA_DICH":
            dl[i]= "N"
        if name == "MA_TA":
            dl[i]= "n"
        if name == "TINH_DICH":
            dl[i]= "B"
        if name == "TINH_TA":
            dl[i]= "b"
        if name == "SI_DICH":
            dl[i]= "A"
        if name == "SI_TA":
            dl[i]= "a"
        if name == "TUONG_DICH":
            dl[i]= "K"
        if name == "TUONG_TA":
            dl[i]= "k"
        if name == "PHAO_DICH":
            dl[i]= "C"
        if name == "PHAO_TA":
            dl[i]= "c"
        if name == "TOT_DICH":
            dl[i]= "P"
        if name == "TOT_TA":
            dl[i]= "p"
        if name == "TRONG":
            dl[i]= "."
        temp = temp.replace("(" + str(89-i) + ")", dl[i])
    return temp

def chess_truoc(dulieuchess):
    dl=dulieuchess
    initial = (
        '               \n'  # 0 -  9
        '               \n'  # 10 - 19
        '               \n'  # 10 - 19
        '   (0)(1)(2)(3)(4)(5)(6)(7)(8)   \n'  # 20 - 29
        '   (9)(10)(11)(12)(13)(14)(15)(16)(17)   \n'  # 40 - 49
        '   (18)(19)(20)(21)(22)(23)(24)(25)(26)   \n'  # 40 - 49
        '   (27)(28)(29)(30)(31)(32)(33)(34)(35)   \n'  # 30 - 39
        '   (36)(37)(38)(39)(40)(41)(42)(43)(44)   \n'  # 50 - 59
        '   (45)(46)(47)(48)(49)(50)(51)(52)(53)   \n'  # 70 - 79
        '   (54)(55)(56)(57)(58)(59)(60)(61)(62)   \n'  # 80 - 89
        '   (63)(64)(65)(66)(67)(68)(69)(70)(71)   \n'  # 70 - 79
        '   (72)(73)(74)(75)(76)(77)(78)(79)(80)   \n'  # 70 - 79
        '   (81)(82)(83)(84)(85)(86)(87)(88)(89)   \n'  # 90 - 99
        '               \n'  # 100 -109
        '               \n'  # 100 -109
        '               \n'  # 110 -119
    )
    temp = initial
    for i in range(0,90):
        name=dl[i]
        namem=''
        if name == "XE_DICH" :
            namem= "r"
        if name == "XE_TA" :
            namem= "R"
        if name == "MA_DICH" :
            namem= "n"
        if name == "MA_TA" :
            namem= "N"
        if name == "TINH_DICH" :
            namem= "b"
        if name == "TINH_TA" :
            namem= "B"
        if name == "SI_DICH" :
            namem= "a"
        if name == "SI_TA" :
            namem= "A"
        if name == "TUONG_DICH" :
            namem= "k"
        if name == "TUONG_TA" :
            namem= "K"
        if name == "PHAO_DICH" :
            namem= "c"
        if name == "PHAO_TA" :
            namem= "C"
        if name == "TOT_DICH" :
            namem= "p"
        if name == "TOT_TA" :
            namem= "P"
        if name == "TRONG" :
            namem= "."
        temp=temp.replace("("+str(i)+")",namem)
    return temp

def chuyendoimatrantadich(matranco):
    temp=0
    dl=matranco
    for i in range(0,30):
        if dl[i]=="TUONG_DO":
            temp=1
            break
    for i in range(0,90):
        name = dl[i]
        if temp==1:
            if name == "XE_DEN":
                dl[i] = "XE_TA"
            if name == "XE_DO":
                dl[i] = "XE_DICH"
            if name == "MA_DEN":
                dl[i] = "MA_TA"
            if name == "MA_DO":
                dl[i] = "MA_DICH"
            if name == "TINH_DEN":
                dl[i] = "TINH_TA"
            if name == "TINH_DO":
                dl[i] = "TINH_DICH"
            if name == "SI_DEN":
                dl[i] = "SI_TA"
            if name == "SI_DO":
                dl[i] = "SI_DICH"
            if name == "TUONG_DEN":
                dl[i] = "TUONG_TA"
            if name == "TUONG_DO":
                dl[i] = "TUONG_DICH"
            if name == "PHAO_DEN":
                dl[i] = "PHAO_TA"
            if name == "PHAO_DO":
                dl[i] = "PHAO_DICH"
            if name == "TOT_DEN":
                dl[i] = "TOT_TA"
            if name == "TOT_DO":
                dl[i] = "TOT_DICH"
        else:
            if name == "XE_DEN":
                dl[i] = "XE_DICH"
            if name == "XE_DO":
                dl[i] = "XE_TA"
            if name == "MA_DEN":
                dl[i] = "MA_DICH"
            if name == "MA_DO":
                dl[i] = "MA_TA"
            if name == "TINH_DEN":
                dl[i] = "TINH_DICH"
            if name == "TINH_DO":
                dl[i] = "TINH_TA"
            if name == "SI_DEN":
                dl[i] = "SI_DICH"
            if name == "SI_DO":
                dl[i] = "SI_TA"
            if name == "TUONG_DEN":
                dl[i] = "TUONG_DICH"
            if name == "TUONG_DO":
                dl[i] = "TUONG_TA"
            if name == "PHAO_DEN":
                dl[i] = "PHAO_DICH"
            if name == "PHAO_DO":
                dl[i] = "PHAO_TA"
            if name == "TOT_DEN":
                dl[i] = "TOT_DICH"
            if name == "TOT_DO":
                dl[i] = "TOT_TA"
    return dl

def fix_move(mang):
    move=mang
    mangmove=[move[0],move[1],move[2],move[3]]
    for x in range(0,4):
        name=mangmove[x]
        if name=='a':
            mangmove[x] = 'i'
        if name=='b':
            mangmove[x] = 'h'
        if name=='c':
            mangmove[x] = 'g'
        if name=='d':
            mangmove[x] = 'f'
        if name=='i':
            mangmove[x] = 'a'
        if name=='h':
            mangmove[x] = 'b'
        if name=='g':
            mangmove[x] = 'c'
        if name=='f':
            mangmove[x] = 'd'
        if name == '0':
            mangmove[x] = '9'
        if name == '9':
             mangmove[x] = '0'
        if name == '1':
            mangmove[x] = '8'
        if name == '8':
            mangmove[x] = '1'
        if name == '2':
            mangmove[x] = '7'
        if name == '7':
            mangmove[x] = '2'
        if name == '3':
            mangmove[x] = '6'
        if name == '6':
            mangmove[x] = '3'
        if name == '4':
            mangmove[x] = '5'
        if name == '5':
            mangmove[x] = '4'
    return mangmove
def sinhmatrantruoc(matranhientai):
    mang=matranhientai
    j=0
    for i in range(0,90):
        if mang[i]=='TRONG':
            j=i
            break
    mang[j]='TUONG_DEN'
    return mang
######################################################################################################################
######################################################################################################################
# CHUONG TRINH CHINH CUA CNN
heightImg =400*6
widthImg = 360*6
img = cv2.imread("theco18.jpg")
#img = cv2.resize(img,(360*3,400*3))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
imgThreshold = preProcess(img)
imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # DRAW ALL DETECTED CONTOURS
tam=cv2.resize(imgContours,(500,500))
#   #### 3. FIND THE BIGGEST COUNTOUR AND USE IT AS SUDOKU
biggest, maxArea = biggestContour(contours)  # FIND THE BIGGEST CONTOUR
if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(img, biggest, -1, (0, 0, 255), 10)  # DRAW THE BIGGEST CONTOUR
    imgtam=cv2.resize(img,(500,600))
   # cv2.imshow("countorbiggest", imgtam)
    pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    tap=cv2.resize(imgWarpColored,(500,500))
   # cv2.imshow("o to nhat",tap)
    boxes = splitBoxes(imgWarpColored)
else:
    boxes=np.array(0,0)
mangco=[]
for i in range(0,90):
    curImg=boxes[i]
    if kiemtradauO(curImg)==0:
        mangco.append("TRONG")
    else:
        imgOriginal_1 = proset_img(curImg)
        imge = np.asarray(imgOriginal_1)
        imgr = cv2.resize(imge, (56, 56))
        imgr = preProcessing(imgr)
        imgr = imgr.reshape(1, 56, 56, 1)
        classIndex = int(model.predict_classes(imgr))
        predictions = model.predict(imgr)
        probVal = np.amax(predictions)
        if probVal > threshold:
            cv2.putText(imgWarpColored, chessname(classIndex) ,nameposition(i),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 200, 255), 4)
            mangco.append(chessname(classIndex))
kq=cv2.resize(imgWarpColored,(720,800))
cv2.imshow("kq",kq)
print(mangco)
cv2.waitKey(0)
#########################################################################################################################
# CHUONG TRINH CHINH CUA MINIMAX
matranhientai=mangco.copy()
matranhientai1 = matranhientai.copy()
matrantruoc = sinhmatrantruoc(matranhientai1)
#####################################################
print("Mời bạn nhập tính năng (1:Tiên đoán nước đi đối phương. 2:Giải thế cờ) : ")
a = input()
if int(a) == 1:
    print("Bạn đã chọn tính năng tiên đoán nước đi của đối phương ")
    print(" Bàn cờ :")
    print(len(matranhientai))
    matrantadichhientai_0 = chuyendoimatrantadich(matranhientai)
    hinhhientai = chess_truoc(matrantadichhientai_0)
    hist_hientai = [Position(hinhhientai, 0)]
    print_pos(hist_hientai[-1])
    matrantadichtruoc = chuyendoimatrantadich(matrantruoc)
    matrantadichhientai = chuyendoimatrantadich(matranhientai)
    hinhcotruoc = chess_truoc(matrantadichtruoc)
    hinhcohientai = chess_hientai(matrantadichhientai)
    hist = [Position(hinhcotruoc, 0)]
    searcher = Searcher()
    print("Đang tính toán nước đi ")
    while True:
        if hist[-1].score <= -MATE_LOWER:
            print("You lost")
            break
        hist.append(Position(hinhcohientai, 0))
        if hist[-1].score <= -MATE_LOWER:
            print("You won")
            break
        start = time.time()
        for _depth, move, score in searcher.search(hist[-1], hist):
            if time.time() - start > THINK_TIME:
                break

        if score == MATE_UPPER:
            print("Checkmate!")
        # The black player moves from a rotated position, so we have to
        # 'back rotate' the move before printing it.

        print("Think depth: {}. Nước đi có thể của đối phương là : {}".format(_depth,render(255 - move[0] - 1) + render(255 - move[1] - 1)))

        hist.append(hist[-1].move(move))
        print_pos1(hist[-1])
        break

if int(a) == 2:
    print("Bạn đã chọn tính năng giải thế cờ ")
    print(" Bàn cờ :")
    matranhientai_n = matranhientai[::-1]
    matrantruoc_n = matrantruoc[::-1]
    matrantadichhientai = chuyendoimatrantadich(matranhientai_n)
    matrantadichtruoc = chuyendoimatrantadich(matrantruoc_n)
    hinhcotruoc = chess_truoc(matrantadichtruoc)
    hinhcohientai = chess_hientai(matrantadichhientai)
    hist = [Position(hinhcotruoc, 0)]
    searcher = Searcher()
    matrantadichhientai = chuyendoimatrantadich(matranhientai)
    # ################## HIEN THỊ TRANG THAI HIEN TAI CHUA XU LI
    hinhhientai = chess_truoc(matrantadichhientai)
    hist_hientai = [Position(hinhhientai, 0)]
    print_pos(hist_hientai[-1])
    print("Đang tính toán nước đi ")
    while True:
        # print(hist)
        if hist[-1].score <= -MATE_LOWER:
            print("You lost")
            break
        hist.append(Position(hinhcohientai, 0))
        if hist[-1].score <= -MATE_LOWER:
            print("You won")
            break
        # Fire up the engine to look for a move.
        start = time.time()
        for _depth, move, score in searcher.search(hist[-1], hist):
            if time.time() - start > THINK_TIME:
                break
        if score == MATE_UPPER:
            print("Checkmate!")
        temp = render(255 - move[0] - 1) + render(255 - move[1] - 1)
        temp = fix_move(temp)
        kq = temp[0] + temp[1] + temp[2] + temp[3]
        print("Think depth: {}. Bạn nên đi nước này : {}".format(_depth, kq))
        hist.append(hist[-1].move(move))
        print_pos2(hist[-1])
        break
