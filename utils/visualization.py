from cProfile import label

import matplotlib.pyplot as plt
import numpy as np


class Visu():
    def __init__(self, env_params):
        self.env_params = env_params
        shape = env_params["shape"]
        self.x_shape = shape["y"]
        self.y_shape = shape["x"]
        n = 0.0001
        self.y_ticks = [[-0.5+i-n, -0.5+i+n] for i in range(self.y_shape+1)]
        self.x_ticks = [[-0.5+i-n, -0.5+i+n] for i in range(self.x_shape+1)]
        self.x_ticks = [item for sublist in self.x_ticks for item in sublist]
        self.y_ticks = [item for sublist in self.y_ticks for item in sublist]
        print("x_ticks", self.x_ticks)
        print("y_ticks", self.y_ticks)
        self.iter = 0
        self.FW_iter = 0
        weights = {}
        self.action_dim = 5
        self.node_size = self.x_shape*self.y_shape
        for t in range(env_params["horizon"]):
            for s in range(self.node_size):
                weights[(t, s)] = 0
        self.subgrad = weights
        self.ret = 111
        self.JPi = dict()
        self.F_M = dict()
        self.JPi_quasi = dict()
        self.JPi_supermodular = dict()
        self.JPi_optimal = None
        self.JPi_dpi_asper_optimal_alpha = None
        self.stationary_pi = None
        self.corner = None

    def path_return(self, ret):
        self.ret = ret

    def mark_subgradient(self, subgrad):
        self.subgrad = subgrad

    def visu_path(self, path):
        f, ax = plt.subplots(figsize=(self.x_shape, self.y_shape))
        # path = [(0, 11), (1, 15), (2, 14), (3, 13), (4, 9)]
        x = []
        y = []

        for cell in path:
            x.append(cell[1] % self.x_shape)
            y.append(int(cell[1]/self.x_shape))

        x_grad = []
        y_grad = []
        list_traj_subgrad = []
        for cell in range(self.x_shape*self.y_shape):
            x_grad.append(cell % self.x_shape)
            y_grad.append(int(cell/self.x_shape))
            cell_grads = [int(self.subgrad.tolist()[cell+h*(self.x_shape*self.y_shape)])
                          for h in range(self.env_params["horizon"])]
            list_traj_subgrad.append(cell_grads)

        print("x", x)
        print("y", y)
        # x = [cell[0] for cell in path]
        # y = [cell[1] for cell in path]
        ax.minorticks_on()
        ax.set_yticks(self.y_ticks, minor=True)
        # ax.yaxis.grid(True, which='minor')
        ax.set_xticks(self.x_ticks, minor=True)
        # ax.xaxis.grid(True, which='minor')
        # plt.grid(axis='both', which='minor')
        ax.grid(b=True, which='minor', axis="both",
                color='k', linestyle='-', linewidth=0.8)
        plt.xlim([-0.5, self.x_shape-0.5])
        plt.ylim([-0.5, self.y_shape-0.5])
        plt.plot(x, y)
        for a, b, c in zip(x_grad, y_grad, list_traj_subgrad):
            plt.text(a, b, str(c), horizontalalignment='center')

        x_noise = [loc + np.random.uniform(-0.1, 0.1, 1).item() for loc in x]
        y_noise = [loc + np.random.uniform(-0.0, 0.0, 1).item() for loc in y]
        plt.plot(x_noise, y_noise, ".", color="red")
        plt.plot(x[0], y[0], "*", color="tab:orange")
        plt.plot(x[-1], y[-1], "*", color="tab:green")
        ax.set_title(
            "Iteration "
            + str(self.iter) +
            " return"
            + str(self.ret)
        )
        plt.savefig("fig"+str(self.iter)+".png")
        self.iter += 1

    def stiener_grid(self, items_loc, path=None, init=0):
        f, ax = plt.subplots(figsize=(self.x_shape, self.y_shape))
        x = []
        y = []
        z = []
        for item in items_loc:
            for cell in items_loc[item]:
                x.append(cell % self.x_shape)
                y.append(int(cell/self.x_shape))
                z.append(item[0])
        if path is not None:
            x_agent = []
            y_agent = []
            t_agent = []
            for t, cell in enumerate(path):
                x_agent.append(cell % self.x_shape)
                y_agent.append(int(cell/self.x_shape))
                t_agent.append(t)
            plt.plot(x_agent, y_agent, color="red")

        # path = [(0, 11), (1, 15), (2, 14), (3, 13), (4, 9)]
        
        ax.minorticks_on()
        ax.set_yticks(self.y_ticks, minor=True)
        # ax.yaxis.grid(True, which='minor')
        ax.set_xticks(self.x_ticks, minor=True)
        # ax.xaxis.grid(True, which='minor')
        # plt.grid(axis='both', which='minor')
        ax.grid(b=True, which='minor', axis="both",
                color='k', linestyle='-', linewidth=0.8)
        plt.xlim([-0.5, self.x_shape-0.5])
        plt.ylim([-0.5, self.y_shape-0.5])
        plt.plot(x, y, ".")
        for a, b, c in zip(x, y, z):
            plt.text(a, b, str(c), fontsize=20,horizontalalignment='center')

        x_init = init % self.x_shape
        y_init = int(init/self.x_shape)
        plt.plot(x_init, y_init , "*", color="tab:green")
        # x_noise = [loc + np.random.uniform(-0.1, 0.1, 1).item() for loc in x]
        # y_noise = [loc + np.random.uniform(-0.0, 0.0, 1).item() for loc in y]
        
        # plt.plot(x[0], y[0], "*", color="tab:orange")
        # plt.plot(x[-1], y[-1], "*", color="tab:green")
        ax.set_title(
            "Iteration "
            + str(self.iter) +
            " return"
            + str(self.ret)
        )
        plt.savefig("fig"+str(self.iter)+".png")
        self.iter += 1
        return plt, f
        


    def visu_path_lb(self, path):
        f, ax = plt.subplots(figsize=(self.x_shape, self.y_shape))
        # path = [(0, 11), (1, 15), (2, 14), (3, 13), (4, 9)]
        x = []
        y = []

        list_traj_subgrad = []
        for cell in path:
            x.append(cell[1] % self.x_shape)
            y.append(int(cell[1]/self.x_shape))
            list_traj_subgrad.append(self.subgrad[cell])

        dict_non_traj = dict()
        for t, s in self.subgrad:
            if s not in dict_non_traj:
                dict_non_traj[s] = []
            dict_non_traj[s].append(self.subgrad[(t, s)])

        x_nontraj = []
        y_nontraj = []
        list_non_traj_subgrad = []
        for states in dict_non_traj:
            x_nontraj.append(states % self.x_shape)
            y_nontraj.append(int(states/self.x_shape))
            list_non_traj_subgrad.append(max(dict_non_traj[states]))

        print("x", x)
        print("y", y)
        # x = [cell[0] for cell in path]
        # y = [cell[1] for cell in path]
        ax.minorticks_on()
        ax.set_yticks(self.y_ticks, minor=True)
        # ax.yaxis.grid(True, which='minor')
        ax.set_xticks(self.x_ticks, minor=True)
        # ax.xaxis.grid(True, which='minor')
        # plt.grid(axis='both', which='minor')
        ax.grid(b=True, which='minor', axis="both",
                color='k', linestyle='-', linewidth=0.8)
        plt.xlim([-0.5, self.x_shape-0.5])
        plt.ylim([-0.5, self.y_shape-0.5])
        plt.plot(x, y)
        for a, b, c in zip(x, y, list_traj_subgrad):
            plt.text(a, b, str(c))
        for a, b, c in zip(x_nontraj, y_nontraj, list_non_traj_subgrad):
            plt.text(a, b, str(c))
        x_noise = [loc + np.random.uniform(-0.1, 0.1, 1).item() for loc in x]
        y_noise = [loc + np.random.uniform(-0.0, 0.0, 1).item() for loc in y]
        plt.plot(x_noise, y_noise, ".", color="red")
        plt.plot(x[0], y[0], "*", color="tab:orange")
        plt.plot(x[-1], y[-1], "*", color="tab:green")
        ax.set_title(
            "Iteration "
            + str(self.iter) +
            " return"
            + str(self.ret)
        )
        plt.savefig("fig"+str(self.iter)+".png")
        self.iter += 1

    def record(self, F_M, Jpi):
        self.F_M[self.iter] = F_M
        self.JPi[self.iter] = Jpi
        return 1

    def recordFW(self, JPi_quasi, idx):
        self.JPi_quasi[idx] = JPi_quasi

    def recordFW_SupMod(self, JPi_supermodular, idx):
        self.JPi_supermodular[idx] = JPi_supermodular

    def plot_recorded(self):
        plt.plot([self.F_M[key] for key in self.F_M],
                 label='multi-ext', color='tab:orange')
        plt.plot([self.JPi[key] for key in self.JPi],
                 label='J_pi', color='tab:blue')
        plt.plot([self.JPi_quasi[key] for key in self.JPi_quasi],
                 label='JPi_quasi', color='tab:purple')
        plt.plot([self.JPi_supermodular[key] for key in self.JPi_supermodular],
                 label='JPi_sup-mod', color='tab:brown')
        plt.axhline(y=self.JPi_optimal, label="J_Pi*", color='tab:green')
        plt.axhline(y=self.JPi_dpi_asper_optimal_alpha, ls='--',
                    label="J_dh_Pi*", color='tab:green')
        plt.xlim([0, self.iter])
        # plt.axhline(y=self.stationary_pi, label="stationary", color='tab:red')
        plt.legend()
        plt.title("start state: " + str(self.corner) + ", (H,S,A)=" +
                  "(" + str(self.env_params["horizon"]) + "," + str(self.node_size) + "," + str(self.action_dim)+")")
        plt.savefig("fig"+str(self.iter)+".png")
        plt.show()
