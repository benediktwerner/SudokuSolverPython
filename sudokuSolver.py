from Board import Board

def solve(board, cell_index=0):
    if cell_index == 81:
        return True
    row = cell_index // 9
    column = cell_index % 9

    if board.isFilled(row, column):
        return solve(board, cell_index+1)

    possible_cell_values = board.getPossibleCellValues(row, column)
    for value in possible_cell_values:
        board.setCell(row, column, value)
        if solve(board, cell_index+1):
            return True

    board.setCell(row, column, 0)
    return False


test_board = [
    [7,0,2,6,0,0,0,0,9],
    [0,0,0,0,1,2,0,0,0],
    [0,3,0,8,0,0,0,6,0],
    [0,0,7,0,0,5,4,0,0],
    [0,0,0,2,7,8,0,0,0],
    [0,0,5,3,0,0,7,0,0],
    [0,9,0,0,0,3,0,5,0],
    [0,0,0,9,8,0,0,0,0],
    [3,0,0,0,0,6,9,0,8]]
b = Board(test_board)
print(b)

solve(b)
print(b)
