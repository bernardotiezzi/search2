"""
Intelligenza Artificiale, CdL Ingegneria Informatica Firenze - Fall 2017

Paolo Frasconi

This module implements the classic 15 puzzle.
"""
from __future__ import print_function
from copy import deepcopy
import sys
import random

from search import Problem
from search import breadth_first_tree_search
from search import breadth_first_graph_search
from search import depth_first_graph_search
from search import depth_first_tree_search
from search import iterative_deepening_search
from search import depth_limited_search
from search import astar_search
from search import greedy_best_first_graph_search
from search import compare_searchers
from search import trace

class PuzzleState:
    """
    The board is NxN so use N=4 for the 15-puzzle, N=5 for
    the 24-puzzle, etc
    The state of the puzzle is simply a permutation of 0..N-1
    The position of the blank (element zero) is stored in r,c
    """
    def __init__(self,board,N,r,c):
        self.board=board
        self.r=r
        self.c=c
        self.N = N
    def __getitem__(self,(r,c)):
        return self.board[r*self.N+c]
    def __setitem__(self,(r,c),val):
        self.board[r*self.N+c]=val
    def move(self,direction):
        ch=deepcopy(self)
        c,r = ch.c,ch.r
        if direction=='left' and self.c != 0:
            ch[(r,c)],ch[(r,c-1)] = self[(r,c-1)],self[(r,c)]
            ch.c = c-1
        elif direction=='right' and c != self.N-1:
            ch[(r,c)],ch[(r,c+1)] = self[(r,c+1)],self[(r,c)]
            ch.c = c+1
        elif direction=='up' and self.r != 0:
            ch[(r,c)],ch[(r-1,c)] = self[(r-1,c)],self[(r,c)]
            ch.r = r-1
        elif direction=='down' and r != self.N-1:
            ch[(r,c)],ch[(r+1,c)] = self[(r+1,c)],self[(r,c)]
            ch.r = r+1
        else:
            return None
        return ch

    def misplaced(self):
        """Misplaced tiles heuristic"""
        blank = self.r*self.N+self.c
        return sum([idx!=val for idx,val in enumerate(self.board) if idx!=blank])

    def manhattan(self):
        """Manhattan distance heuristic"""
        m=0
        blank = self.r*self.N+self.c
        for index,value in enumerate(self.board):
            if index != blank and index != value:
                r = index // self.N
                c = index % self.N
                rr = value // self.N
                cc = value % self.N
                # print('misplaced',value,rr,r,cc,c)
                m += abs(r-rr) + abs(c-cc)
        assert(m>=0)
        return m
# Exercise: improve using linear conflicts

# Exercise: add pattern database heuristic
    def __str__(self):
        """Serialize the state in a human-readable form"""
        s = ''
        for r in xrange(self.N):
            for c in xrange(self.N):
                if self[(r,c)]>0:
                    s += '%3d' % self[(r,c)]
                else:
                    s += '   '
            s += '\n'
        return s
    def __repr__(self):
        return self.__str__()


class Puzzle(Problem):
    """Base class - For 8-puzzle use Puzzle(3) -- a 3x3 grid"""
    def __init__(self, N, seed,scrambles=10):
        self.N = N
        self.actions = ['left','right','up','down']
        self.make_initial_state(seed,scrambles)

    def make_initial_state(self,seed,scrambles):
        """
        To ensure a solution exists, start from the goal and scramble
        it applying a random sequence of actions. An alternative is to
        use the permutation parity property of the puzzle but using
        the scrambling we can roughly control the depth of the
        solution and thus limit CPU time for demonstration
        """
        seen = {}
        ns=0
        x = range(self.N*self.N)
        for r in range(self.N):
            for c in range(self.N):
                if x[r*self.N+c]==0:
                    row,col=r,c
        self.initial = PuzzleState(x,self.N,row,col)
        R = random.Random()
        R.seed(seed)
        while ns<scrambles:
            index = R.randint(0,len(self.actions)-1)
            a = self.actions[index]
            nexts = self.initial.move(a)
            if nexts is not None:
                serial = nexts.__str__()
                if serial not in seen:
                    seen[serial] = True
                    self.initial = nexts
                    ns += 1
        print('Problem:', self.__doc__, 'Initial state:')
        print(self.initial)
        print('==============')

    def successor(self, state):
        """Legal moves (blank moves left, right, up,
        down). Implemented as a generator"""
        for action in self.actions:
            nexts = state.move(action)
            if nexts is not None:
                yield (action,nexts)

    def goal_test(self, state):
        """For simplicity blank on top left"""
        return state.board==range(self.N*self.N)

    def h(self,node):
        """No heuristic. A* becomes uniform cost in this case"""
        return 0

class PuzzleMisplaced(Puzzle):
    """Misplaced heuristic"""
    def h(self,node):
        return node.state.misplaced()

class PuzzleManhattan(Puzzle):
    """Manhattan heuristic"""
    def h(self,node):
        return node.state.manhattan()

class PuzzleManhattanOver(Puzzle):
    """Inadmissible Manhattan"""
    def h(self,node):
        return 1.2*node.state.manhattan()

# ____________________________________________________________________________
# Some predefined demos

def blind_puzzle(seed, scrambles):
    compare_searchers(problems=[Puzzle(3,seed,scrambles)],
                      header=['Searcher','8-puzzle'],
                      searchers=[breadth_first_tree_search,
                                 breadth_first_graph_search,
                                 depth_first_graph_search,
                                 iterative_deepening_search,
                                 depth_limited_search])

def crazy(seed, scrambles):
    compare_searchers(problems=[Puzzle(3,seed,scrambles)],
                      header=['Searcher','8-puzzle'],
                      searchers=[depth_first_tree_search])

def informed_puzzle(seed, scrambles):
    compare_searchers(problems=[Puzzle(3,seed,scrambles),
                      PuzzleMisplaced(3,seed,scrambles),
                      PuzzleManhattan(3,seed,scrambles)],
                      header=['Searcher','8-puzzle h=0',
                      '8-puzzle misplaced', '8-puzzle Manhattan'],
                      searchers=[astar_search])

def informed_puzzle15(seed,scrambles):
    compare_searchers(problems=[PuzzleMisplaced(4,seed,scrambles),
                      PuzzleManhattan(4,seed,scrambles)],
                      header=['Searcher','15-puzzle misplaced',
                      '15-puzzle Manhattan'],
                      searchers=[astar_search,greedy_best_first_graph_search])


def informed_puzzle_over(seed,scrambles):
    compare_searchers(problems=[PuzzleManhattan(3,seed,scrambles),
                      PuzzleManhattanOver(3,seed,scrambles)],
                      header=['Searcher','8-puzzle Manhattan',
                      '8-puzzle Manhattan over'],
                      searchers=[astar_search,greedy_best_first_graph_search])

def informed_puzzle15_over(seed,scrambles):
    compare_searchers(problems=[PuzzleManhattan(4,seed,scrambles),
                      PuzzleManhattanOver(4,seed,scrambles)],
                      header=['Searcher','15-puzzle Manhattan',
                      '15-puzzle Manhattan over'],
                      searchers=[astar_search,greedy_best_first_graph_search])