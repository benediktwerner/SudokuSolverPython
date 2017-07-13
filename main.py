import detector
import sudokuSolver

sudoku = detector.detect_sudoku("sudoku4.jpg")
print(sudoku)
print(sudokuSolver.solve(sudoku))
print(sudoku)
