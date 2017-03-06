class Board:
    def __init__(self, size, board=None):
        self.size = size
        self.board = [[0]*size]*size if board == None else board

    def isFilled(self, row, col):
        return self.board[row][col] != 0

    def setCell(self, row, column, value):
        self.board[row][column] = value

    def getPossibleCellValues(self, row, column):
        possible_values = set(range(1,self.size+1))
        possible_values -= self.getValuesUsedInRow(row)
        possible_values -= self.getValuesUsedInColumn(column)
        possible_values -= self.getValuesUsedInBox(row, column)
        return possible_values

    def getValuesUsedInRow(self, row):
        values = set()
        for column in range(self.size):
            values.add(self.board[row][column])
        return values

    def getValuesUsedInColumn(self, column):
        values = set()
        for row in range(self.size):
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
        for row in range(self.size):
            for column in range(self.size):
                string += str(self.board[row][column]) if self.board[row][column] != 0 else "."
                if column in (2, 5):
                    string += " | "
            string += "\n"
            if row in (2,5):
                string += "----+-----+----\n"
        return string
