from os.path import join
from os import getcwd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Path("results").mkdir(exist_ok=True)


class Saver:
    def __init__(self, filename):
        self.filename = filename
        self.dir = join(getcwd(), "results")
        self.save_path = join(self.dir, self.filename)
        self.fileobj = open(self.save_path, "w")
        self.fileobj.write("Episodes, Test_Sample_Rate, Total_Avg_Rate\n")

    def save(self, data):
        str_data = list(map(str, data))
        self.fileobj.write(",".join(str_data) + "\n")

    def close(self):
        self.fileobj.close()


class Plotter:
    def __init__(self, filename):
        self.filename = filename
        self.dir = join(getcwd(), "results")
        self.save_path = join(self.dir, self.filename)
        self.df = pd.read_csv(self.save_path)
        pass

    def plot(self):
        # multiple line plot
        x, y1, y2 = self.df.keys()
        plt.plot(x, y1, data=self.df, color='skyblue')
        plt.plot(x, y2, data=self.df, marker='', color='olive', linewidth=2)
        # plt.plot('Episodes', 'y3', data=self.df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
        plt.xlabel = "Episodes"
        plt.ylabel = "Rate of Success (%)"
        plt.legend()
        plt.show()



def main():
    learner_types = ["QLearner", "Sarsa", "DynaQ"]
    board_sizes = ["16"]
    dir = join(getcwd(), "results")
    for learner in learner_types:
        for size in board_sizes:
            filename = "{}_{}.csv".format(learner, size)
            plotter = Plotter(filename)
            save_path = join(dir, filename)
            df = pd.read_csv(save_path)
            x, y1, y2 = df.keys()
            # plt.plot(x, y1, data=df, color='skyblue')
            plt.plot(x, y2, data=df, marker='', linewidth=2,label=learner)
    plt.xlabel = "Episodes"
    plt.ylabel = "Rate of Success (%)"
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
