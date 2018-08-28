import search, puzzle
searcher = search.astar_search
problem = puzzle.PuzzleManhattan(4, seed=125, scrambles=45)
solution = searcher(problem)
path = solution.path()
path.reverse()
print path
