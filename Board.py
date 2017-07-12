def check_board(board):
    if len(board) != 9:
        raise "Board height is not 9"
    for row in board:
        if len(row) != 9:
            raise "Board width is not 9"

class Board:
    def __init__(self, board):
        check_board(board)
        self.board = board

    def isFilled(self, row, col):
        return self.board[row][col] != 0

    def setCell(self, row, column, value):
        self.board[row][column] = value

    def getPossibleCellValues(self, row, column):
        possible_values = set((1,2,3,4,5,6,7,8,9))
        possible_values -= self.getValuesUsedInRow(row)
        possible_values -= self.getValuesUsedInColumn(column)
        possible_values -= self.getValuesUsedInBox(row, column)
        return possible_values

    def getValuesUsedInRow(self, row):
        values = set()
        for column in range(9):
            values.add(self.board[row][column])
        return values

    def getValuesUsedInColumn(self, column):
        values = set()
        for row in range(9):
            values.add(self.board[row][column])
        return values

    def getValuesUsedInBox(self, row, column):
        values = set()
        box_row = row - row % 3
        box_column = column - column % 3
        for row_index in range(3):
            for column_index in range(3):
                values.add(self.board[box_row + row_index][box_column + column_index])
        return values

    def __str__(self):
        string = ""
        for row in range(9):
            for column in range(9):
                string += str(self.board[row][column]) if self.board[row][column] != 0 else "."
                if column in (2, 5):
                    string += " | "
            string += "\n"
            if row in (2,5):
                string += "----+-----+----\n"
        return string
