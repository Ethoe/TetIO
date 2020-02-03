import numpy as np
from mss import mss
from pynput.keyboard import Key, Controller
import matplotlib.pyplot as plot
import pytesseract
import time
import pywinauto
import copy
import random


class Network:

    def __init__(self, input_num, hidden_num, outputs_num):
        self.inputs_num = input_num
        self.hidden_layer_num = hidden_num
        self.outputs_num = outputs_num

        w1 = 2 * np.random.rand(self.hidden_layer_num, self.inputs_num) - 1
        w2 = 2 * np.random.rand(self.outputs_num, self.hidden_layer_num) - 1
        self.weights = [w1, w2]

        b1 = 2 * np.random.rand(self.hidden_layer_num, 1) - 1
        b2 = 2 * np.random.rand(self.outputs_num, 1) - 1
        self.bias = [b1, b2]

    @staticmethod
    def sig(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def feedforward(self, inputs):
        for layer in range(2):
            if layer == 0:
                node_in = inputs
            else:
                node_in = next_layer
            if layer == 0:
                z = np.add(self.weights[layer].dot(node_in), self.bias[layer][0])
                next_layer = self.relu(z)
            else:
                z = np.add(self.weights[layer].dot(node_in), self.bias[layer][0])
                next_layer = self.softmax(z)
        return next_layer

    @staticmethod
    def genetics(generation_matrix, high_percentage, random_percentage, mutation_rate, gen_size):
        new_generation = [[]]
        new_generation[0] = generation_matrix[0]
        high_fitness = int(high_percentage * gen_size)
        rand_saved = int(random_percentage * gen_size)
        saved = np.random.choice(gen_size - high_fitness, rand_saved)
        for parent in range(high_fitness):
            new_generation.append(generation_matrix[parent + 1])
            new_generation[-1][1] = 0
        for save in saved:
            new_generation.append(generation_matrix[save + high_fitness + 1])
            new_generation[-1][1] = 0
        for parent in range(high_fitness + rand_saved):
            for i in range(1000):
                for j in range(215):
                    if mutation_rate >= np.random.random_sample():
                        generation_matrix[parent + 1][0].weights[0][i][j] += 2 * np.random.random_sample() - 1
            for i in range(37):
                for j in range(1000):
                    if mutation_rate >= np.random.random_sample():
                        generation_matrix[parent + 1][0].weights[1][i][j] += 2 * np.random.random_sample() - 1
            for i in range(1000):
                if mutation_rate >= np.random.random_sample():
                    generation_matrix[parent + 1][0].bias[0][i] += 2 * np.random.random_sample() - 1
            for i in range(37):
                if mutation_rate >= np.random.random_sample():
                    generation_matrix[parent + 1][0].bias[1][i] += 2 * np.random.random_sample() - 1
        for new_child in range(gen_size - (high_fitness + rand_saved)):
            parents = np.random.choice(high_fitness + rand_saved, 2)
            child = copy.deepcopy(generation_matrix[parents[0] + 1][0])
            for i in range(1000):
                for j in range(215):
                    child.weights[0][i][j] = random.choice([generation_matrix[parents[0] + 1][0].weights[0][i][j],
                                                            generation_matrix[parents[1] + 1][0].weights[0][i][j]])
            for i in range(37):
                for j in range(1000):
                    child.weights[1][i][j] = random.choice([generation_matrix[parents[0] + 1][0].weights[1][i][j],
                                                            generation_matrix[parents[1] + 1][0].weights[1][i][j]])
            for i in range(1000):
                child.bias[0][i] = random.choice([generation_matrix[parents[0] + 1][0].bias[0][i],
                                                  generation_matrix[parents[1] + 1][0].bias[0][i]])
            for i in range(37):
                child.bias[1][i] = random.choice([generation_matrix[parents[0] + 1][0].bias[1][i],
                                                  generation_matrix[parents[1] + 1][0].bias[1][i]])
            new_generation.append([child, 0])
        return np.array(new_generation)


class Jstris:

    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
        self.board_region = {"top": 400, "left": -1008, "width": 240, "height": 480}
        self.queue_region = {"top": 424, "left": -715, "width": 24, "height": 264}
        self.lines_region = {"top": 905, "left": -913, "width": 25, "height": 20}
        self.board_matrix = np.zeros((20, 10)).astype(int)
        self.previous = np.zeros((20, 10)).astype(int)
        self.lines = 0
        self.pieces = 0
        self.keypresses = 0
        self.current_piece = 0
        self.next_piece = 0
        self.game_dead = 0
        self.J, self.L, self.S, self.Z, self.T, self.O, self.I = (0, 1, 2, 3, 4, 5, 6)
        HD, R, DR, L, DL, CW, CCW, CCCW = (0, 1, 2, 3, 4, 5, 6, 7)
        self.finesse_moves = [[HD], [DL, HD], [DL, R, HD], [L, HD], [R, HD], [DR, L, HD], [DR, HD], [CCW, DL, HD],
                              [DL, CCW, HD], [DL, CW, HD], [L, CCW, HD], [CCW, HD], [CW, HD], [R, CW, HD],
                              [DR, CCW, HD], [DR, CW, HD], [CCW, DR, HD], [L, L, HD], [R, R, HD], [CW, DL, HD],
                              [L, L, CW, HD], [L, CW, HD], [R, R, CW, HD], [DR, L, CW, HD], [DL, CCCW, HD],
                              [L, L, CCCW, HD], [L, CCCW, HD], [CCCW, HD], [R, CCCW, HD], [R, R, CCCW, HD],
                              [DR, L, CCCW, HD], [DR, CCCW, HD], [L, L, CCW, HD], [R, CCW, HD], [R, R, CCW, HD],
                              [DR, L, CCW, HD], [CW, DR, HD]]

    def reset(self):
        self.lines = 0
        self.pieces = 0
        self.keypresses = 0
        self.game_dead = 0
        self.board_matrix = np.zeros((20, 10)).astype(int)
        self.previous = np.zeros((20, 10)).astype(int)
        self.current_piece = 0
        self.next_piece = 0

    def sample_queue(self):
        with mss() as sct:
            queue = np.array(sct.grab(self.queue_region))
            piece = [queue[12][12], queue[36][12]]
            if piece[0][0] == 0 and piece[0][1] == 0 and piece[0][2] == 0:
                if piece[1][0] == 198:
                    self.next_piece = self.J
                elif piece[1][0] == 2:
                    self.next_piece = self.L
            else:
                if piece[0][0] == 2:
                    self.next_piece = self.O
                elif piece[0][0] == 55:
                    self.next_piece = self.Z
                elif piece[0][0] == 138:
                    self.next_piece = self.T
                elif piece[0][0] == 215:
                    self.next_piece = self.I
                elif piece[0][0] == 1:
                    self.next_piece = self.S

    def sample_board(self):
        with mss() as sct:
            board = np.array(sct.grab(self.board_region))
            double_previous = copy.deepcopy(self.previous)
            self.previous = copy.deepcopy(self.board_matrix)
            # Shape is 480 by 240
            for x in range(20):
                for y in range(10):
                    pixel = board[12 + 24 * x][12 + 24 * y]
                    if pixel[0] == 1 or pixel[0] == 198 or pixel[0] == 2 or pixel[0] == 55 or pixel[0] == 138 or \
                            pixel[0] == 215:
                        if pixel[0] == 106:
                            self.game_dead = 1
                        self.board_matrix[x][y] = 1
                    else:
                        self.board_matrix[x][y] = 0
            self.board_matrix = self.board_matrix.astype(int)
            if (self.previous == self.board_matrix).all() and (double_previous == self.previous).all():
                self.game_dead = 1

    def sample_lines(self):
        with mss() as sct:
            lines = np.array(sct.grab(self.lines_region))
            try:
                self.lines = 40 - int(pytesseract.image_to_string(lines, config='-psm 6'))
            except:
                self.lines = 0

    def move(self, moves):
        keyboard = Controller()
        for move in moves:
            time.sleep(.005)
            if move == 0:
                keyboard.press(Key.space)
                keyboard.release(Key.space)
            elif move == 1:
                keyboard.press(Key.right)
                keyboard.release(Key.right)
            elif move == 2:
                keyboard.press(Key.right)
                time.sleep(.010)
                keyboard.release(Key.right)
            elif move == 3:
                keyboard.press(Key.left)
                keyboard.release(Key.left)
            elif move == 4:
                keyboard.press(Key.left)
                time.sleep(.010)
                keyboard.release(Key.left)
            elif move == 5:
                keyboard.press(Key.up)
                keyboard.release(Key.up)
            elif move == 6:
                keyboard.press('z')
                keyboard.release('z')
            else:
                keyboard.press(Key.up)
                keyboard.release(Key.up)
                keyboard.press(Key.up)
                keyboard.release(Key.up)
                self.keypresses += 1
        self.keypresses += 1

    def key(self, finesse):
        self.move(self.finesse_moves[finesse])
        self.pieces += 1

    def kpp(self):
        return self.keypresses / self.pieces

    @staticmethod
    def line_fitness(x):
        seg = (0.2 * x - 11.2)
        return (-(seg * seg) + 3.64) * (-np.exp(0.2 * x - 6.2))

    def fitness_sig(self):
        return 1 / (1 + np.exp(-14.5 + 5 * self.kpp()))

    def fitness(self):
        self.sample_lines()
        # return (self.lines + self.line_fitness(self.pieces * (2 / 5))) * self.fitness_sig()
        return (self.line_fitness(self.lines) * 10) + self.line_fitness(self.pieces * (2/5))  # ) * self.fitness_sig()

    @staticmethod
    def piece_array(mino):
        array = [0] * 7
        array[mino] = 1
        return np.array(array)

    def play_game(self, nn):
        self.reset()
        app = pywinauto.Application().connect(title="Jstris - Google Chrome")
        app.JstrisChrome.set_focus()
        keyboard = Controller()
        keyboard.press(Key.f4)
        keyboard.release(Key.f4)
        self.sample_queue()
        self.current_piece = self.next_piece
        time.sleep(1.90)
        while not self.game_dead:
            self.sample_board()
            self.sample_queue()
            inputs = np.append(self.board_matrix, self.piece_array(self.current_piece))
            inputs = np.append(inputs, self.piece_array(self.next_piece)).reshape(214, 1)
            inputs = np.append(inputs, self.pieces / 100)
            choice = np.argmax(nn.feedforward(inputs))
            self.key(choice)
            self.current_piece = self.next_piece
            time.sleep(.01)
        self.pieces -= 3
        if self.pieces <= 0:
            self.pieces = 1
        return self.fitness()


class TetrisAI:
    def __init__(self, generation_size, generation_num, generate_new):
        self.generation_size = generation_size
        self.generations = generation_num
        self.tetris = Jstris()
        self.generate_new = generate_new
        if self.generate_new:  # creates generation_size new nn and makes that the generation array
            self.generation = [[1, 1000.0]]
            for net in range(self.generation_size):
                current_net = Network(215, 1000, 37)
                self.generation.append([current_net, 0])
            self.generation = np.array(self.generation)
        else:
            self.generation = np.array(np.load('generation.npy'))
            self.generation_size = self.generation.shape[0] - 1

    def learn(self, high_fitness_percentage, random_survival_percentage, random_mutation_rate, avg_games):
        xs = []
        ys = []
        plot.ion()
        fig = plot.figure()
        ax = fig.add_subplot(111)

        for gen in range(self.generations):
            print("Generation  " + str(self.generation[0][0]))
            fit_avg = 0
            piece_avg = 0
            for child in range(self.generation_size):
                average = 0
                for item in range(avg_games):
                    average += self.tetris.play_game(self.generation[child + 1][0])
                print('Child ' + str(child) + '/' + str(self.generation_size) +
                      ' Pieces: ' + str(self.tetris.pieces) +
                      ' Fitness: ' + str(average / avg_games))
                fit_avg += average / avg_games
                piece_avg += self.tetris.pieces
                self.generation[child + 1][1] = (average / avg_games)
            print('Average Fitness: ' + str(fit_avg / self.generation_size) +
                  " Average Pieces: " + str(piece_avg / self.generation_size))
            self.generation = self.generation[self.generation[:, 1].argsort()[::-1]]
            self.generation = Network.genetics(self.generation, high_fitness_percentage, random_survival_percentage,
                                               random_mutation_rate, self.generation_size)
            ys.append(fit_avg / self.generation_size)
            xs.append(self.generation[0][0])
            ax.plot(xs, ys, 'k')
            fig.canvas.draw()
            self.generation[0][0] += 1
            np.save('generation', self.generation)


tetris = TetrisAI(100, 200, False)
tetris.learn(.15, .00, .05, 1)
