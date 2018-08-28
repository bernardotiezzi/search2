"""Search (Chapters 3-4)

The way to use this code is to subclass Problem to create a class of problems,
then create problem instances and solve them with calls to the various search
functions."""
from __future__ import print_function
from __future__ import generators

from utils import FIFOQueue,PriorityQueue,Stack,infinity,memoize,name,print_table,update
import sys
import time
import os
import psutil

# ______________________________________________________________________________

class Problem:
    """The abstract class for a formal problem.  You should subclass this and
    implement the method successor, and possibly __init__, goal_test, and
    path_cost. Then you will create instances of your subclass and solve them
    with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def successor(self, state):
        """Given a state, return a sequence of (action, state) pairs reachable
        from this state. If there are many successors, consider an iterator
        that yields the successors one at a time, rather than building them
        all at once. Iterators will work fine within the framework."""
        pass # abstract

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal, as specified in the constructor. Implement this
        method if checking against a single self.goal is not enough."""
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

# ______________________________________________________________________________

class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Create a search tree Node, derived from a parent by an action."
        update(self, state=state, parent=parent, action=action,
               path_cost=path_cost, depth=0)
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        """(pf) Modified to display depth, f and h"""
        if hasattr(self,'f'):
            return "<Node: f=%d, depth=%d, h=%d\n%s>" % (self.f,
                                                         self.depth,
                                                         self.h,
                                                         self.state)
        else:
            return "<Node: depth=%d\n%s>" % (self.depth,self.state)

    def path(self):
        "Create a list of nodes from the root to this node."
        x, result = self, [self]
        while x.parent:
            result.append(x.parent)
            x = x.parent
        return result

    def expand(self, problem):
        "Return a list of nodes reachable from this node. [Fig. 3.8]"
        return [Node(next_state, self, action,
                     problem.path_cost(self.path_cost, self.state, action, next_state))
                for (action, next_state) in problem.successor(self.state)]


# ______________________________________________________________________________
# Uninformed Search algorithms

def tree_search(problem, fringe):
    """Search through the successors of a problem to find a goal.
    The argument fringe should be an empty queue.
    Don't worry about repeated paths to a state. [Fig. 3.8].
    (pf) modified to monitor depth and memory usage"""
    fringe.append(Node(problem.initial))
    max_depth=0
    while fringe:
        node = fringe.pop()
        # Print some information about search progress
        if node.depth>max_depth:
            max_depth=node.depth
            if max_depth<50 or max_depth % 1000 == 0:
                pid = os.getpid()
                py = psutil.Process(pid)
                memoryUse = py.memory_info()[0]/1024/1024
                print('Reached depth',max_depth,
                      'Open len', len(fringe),
                      'Memory used (MBytes)', memoryUse)

        if problem.goal_test(node.state):
            return node
        fringe.extend(node.expand(problem))
    return None

def breadth_first_tree_search(problem):
    "Search the shallowest nodes in the search tree first. [p 74]"
    return tree_search(problem, FIFOQueue())

def depth_first_tree_search(problem):
    "Search the deepest nodes in the search tree first. [p 74]"
    return tree_search(problem, Stack())

def graph_search(problem, fringe):
    """Search through the successors of a problem to find a goal.
    The argument fringe should be an empty queue.
    If two paths reach a state, only use the best one. [Fig. 3.18]"""
    closed = {}
    fringe.append(Node(problem.initial))
    max_depth=0
    while fringe:
        node = fringe.pop()
        # Print some information about search progress
        if node.depth>max_depth:
            max_depth=node.depth
            if max_depth<50 or max_depth % 1000 == 0:
                pid = os.getpid()
                py = psutil.Process(pid)
                memoryUse = py.memory_info()[0]/1024/1024
                print('Reached depth',max_depth,
                      'Open len', len(fringe),
                      'Memory used (MBytes)', memoryUse)

        if problem.goal_test(node.state):
            return node
        serial = node.state.__str__()
        if serial not in closed:
            closed[serial] = True
            fringe.extend(node.expand(problem))
    return None

def breadth_first_graph_search(problem):
    "Search the shallowest nodes in the search tree first. [p 74]"
    return graph_search(problem, FIFOQueue())

def depth_first_graph_search(problem):
    "Search the deepest nodes in the search tree first. [p 74]"
    return graph_search(problem, Stack())

def depth_limited_search(problem, limit=10):
    "[Fig. 3.12]"
    def recursive_dls(node, problem, limit):
        cutoff_occurred = False
        if problem.goal_test(node.state):
            return node
        elif node.depth == limit:
            return 'cutoff'
        else:
            for successor in node.expand(problem):
                result = recursive_dls(successor, problem, limit)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result != None:
                    return result
        if cutoff_occurred:
            return 'cutoff'
        else:
            return None
    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initial), problem, limit)

def iterative_deepening_search(problem):
    "[Fig. 3.13]"
    for depth in xrange(sys.maxint):
        result = depth_limited_search(problem, depth)
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0]/1024/1024
        print('end depth_limited_search at depth', depth, 'mem (GBytes)', memoryUse)
        if result is not 'cutoff':
            return result

# ______________________________________________________________________________
# Informed (Heuristic) Search

def best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have depth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    return graph_search(problem, PriorityQueue(min, f))

# greedy_best_first_graph_search = best_first_graph_search
    # Greedy best-first search is accomplished by specifying f(n) = h(n).
def greedy_best_first_graph_search(problem,h=None):
    h = h or problem.h
    h = memoize(h, 'h')
    def f(n):
        return h(n)
    return best_first_graph_search(problem, f)

def astar_search(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search.
    Uses the pathmax trick: f(n) = max(f(n), g(n)+h(n))."""
    h = h or problem.h
    h = memoize(h, 'h')
    def f(n):
        return max(getattr(n, 'f', -infinity), n.path_cost + h(n))
    return best_first_graph_search(problem, f)

# Exercise: implement weighted heuristic f = lambda*g + (1-lambda)*h
# Exercise: Implement IDA*

# ______________________________________________________________________________
# Peter Norvig - With small changes/additions
# Code to compare searchers on various problems.

class InstrumentedProblem(Problem):
    """Delegates to a problem, and keeps statistics."""

    def __init__(self, problem):
        self.problem = problem
        self.succs = self.goal_tests = self.states = 0
        self.found = None

    def successor(self, state):
        "Return a list of (action, state) pairs reachable from this state."
        result = [r for r in self.problem.successor(state)]
        self.succs += 1
        self.states += len(result)
        return result

    def goal_test(self, state):
        "Return true if the state is a goal."
        self.goal_tests += 1
        result = self.problem.goal_test(state)
        if result:
            self.found = state
        return result

    def __getattr__(self, attr):
        if attr in ('succs', 'goal_tests', 'states'):
            return self.__dict__[attr]
        else:
            return getattr(self.problem, attr)

    def __repr__(self):
        return '#exp=%8d #gt=%8d #st=%8d cost=%6d' % (self.succs, self.goal_tests,
                                                      self.states, self.solution_cost)


def trace(problem, searcher):
    p = InstrumentedProblem(problem)
    solution = searcher(p)
    path=solution.path()
    path.reverse()
    print(path)
    problem.solution_cost = solution.path_cost
    print(p)

def compare_searchers(problems, header, searchers=[breadth_first_tree_search,
                      breadth_first_graph_search, depth_first_graph_search,
                      iterative_deepening_search, depth_limited_search,
                      astar_search]):
    def do(searcher, problem):
        if sys.stdin.isatty():
            raw_input('Press Enter to run %s on %s' % (name(searcher),name(problem)))
        else:
            print('Running %s on %s' % (name(searcher),name(problem)))
        p = InstrumentedProblem(problem)
        start = time.time()
        solution = searcher(p)
        elapsed = time.time() - start
        print('Elapsed time:', elapsed, 'seconds')
        if isinstance(solution,Node):
            path=solution.path()
            print('**************** Solution:')
            if len(path)<50:
                path.reverse()
                print(path)
            else:
                print('len(path)=',len(path),'too long, print suppressed')
            p.solution_cost = solution.path_cost
            print()
        else: # search failed, return a negative value
            p.solution_cost = -1
            print('Search failed')
        return p
    # table = [[name(s)] + [do(s, p) for p in problems] for s in searchers]
    # print_table(table, header, sep='|')
    results={(p,s):do(s,p) for p in problems for s in searchers}
    for p in problems:
        print('Problem:', name(p))
        for s in searchers:
            print('Searcher:%30s' % name(s), results[(p,s)])
    print('#exp: number of expanded nodes')
    print('#gt:  number of goal tests')
    print('#st:  number of generated states')

# Exercise: compute and compare penetrance and effective branching factor