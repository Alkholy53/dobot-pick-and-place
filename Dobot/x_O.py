import tkinter as tk
from tkinter import messagebox
import math

class TicTacToe:
    def _init_(self):
        self.window = tk.Tk()
        self.window.title("Tic-Tac-Toe")
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.current_player = 'O'  # Player is 'O', AI is 'X'
        self.create_board()
        self.window.mainloop()

    def create_board(self):
        for i in range(3):
            for j in range(3):
                button = tk.Button(
                    self.window, text=' ', font=('Arial', 24), height=2, width=5,
                    command=lambda row=i, col=j: self.player_move(row, col)
                )
                button.grid(row=i, column=j)
                self.buttons[i][j] = button

    def player_move(self, row, col):
        if self.board[row][col] == ' ' and self.current_player == 'O':
            self.board[row][col] = 'O'
            self.buttons[row][col].config(text='O', state=tk.DISABLED)
            if self.check_winner():
                messagebox.showinfo("Tic-Tac-Toe", "You win!")
                self.reset_board()
            elif self.is_full():
                messagebox.showinfo("Tic-Tac-Toe", "It's a draw!")
                self.reset_board()
            else:
                self.current_player = 'X'
                self.ai_move()

    def ai_move(self):
        move = self.find_best_move()
        if move:
            row, col = move
            self.board[row][col] = 'X'
            self.buttons[row][col].config(text='X', state=tk.DISABLED)
            if self.check_winner():
                messagebox.showinfo("Tic-Tac-Toe", "AI wins!")
                self.reset_board()
            elif self.is_full():
                messagebox.showinfo("Tic-Tac-Toe", "It's a draw!")
                self.reset_board()
            else:
                self.current_player = 'O'

    def check_winner(self):
        # Check rows, columns, and diagonals
        for row in self.board:
            if row[0] == row[1] == row[2] and row[0] != ' ':
                return True
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] and self.board[0][col] != ' ':
                return True
        if self.board[0][0] == self.board[1][1] == self.board[2][2] and self.board[0][0] != ' ':
            return True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] and self.board[0][2] != ' ':
            return True
        return False

    def is_full(self):
        for row in self.board:
            if ' ' in row:
                return False
        return True

    def minimax(self, depth, is_maximizing):
        winner = self.get_winner()
        if winner == 'X':
            return 10 - depth
        elif winner == 'O':
            return depth - 10
        elif self.is_full():
            return 0

        if is_maximizing:
            max_eval = -math.inf
            for i in range(3):
                for j in range(3):
                    if self.board[i][j] == ' ':
                        self.board[i][j] = 'X'
                        eval = self.minimax(depth + 1, False)
                        self.board[i][j] = ' '
                        max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = math.inf
            for i in range(3):
                for j in range(3):
                    if self.board[i][j] == ' ':
                        self.board[i][j] = 'O'
                        eval = self.minimax(depth + 1, True)
                        self.board[i][j] = ' '
                        min_eval = min(min_eval, eval)
            return min_eval

    def find_best_move(self):
        best_val = -math.inf
        best_move = None

        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    self.board[i][j] = 'X'
                    move_val = self.minimax(0, False)
                    self.board[i][j] = ' '
                    if move_val > best_val:
                        best_val = move_val
                        best_move = (i, j)
        return best_move

    def get_winner(self):
        # Check rows, columns, and diagonals for a winner
        for row in self.board:
            if row[0] == row[1] == row[2] and row[0] != ' ':
                return row[0]
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] and self.board[0][col] != ' ':
                return self.board[0][col]
        if self.board[0][0] == self.board[1][1] == self.board[2][2] and self.board[0][0] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] and self.board[0][2] != ' ':
            return self.board[0][2]
        return None

    def reset_board(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(text=' ', state=tk.NORMAL)
        self.current_player = 'O'

# Start the game
TicTacToe()