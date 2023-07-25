import multiprocessing
import os
import random
from copy import copy
from datetime import datetime
from operator import itemgetter

import gpytorch
from botorch.models import SingleTaskGP
from gpytorch.kernels import ScaleKernel, MaternKernel
import dill as pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import yaml
from scipy import optimize as opt
from sympy import Matrix, MatrixSymbol, derive_by_array, symarray
# from environments.gorilla.openweathermap.KGS_environment import (GridFunction, get_gorillas_density,
#                                             get_jungle_weather)

def horizon_grid_world_graph(world_size, H):
    """Create a graph that represents a grid world.
    In the grid world there are four actions, (1, 2, 3, 4), which correspond
    to going (up, right, down, left) in the x-y plane. The states are
    ordered so that `np.arange(np.prod(world_size)).reshape(world_size)`
    corresponds to a matrix where increasing the row index corresponds to the
    x direction in the graph, and increasing y index corresponds to the y
    direction.
    Parameters
    ----------
    world_size: tuple
        The size of the grid world (rows, columns)
    Returns
    -------
    graph: nx.DiGraph()
        The directed graph representing the grid world.
    """
    nodes = np.arange(np.prod(world_size))
    grid_nodes = nodes.reshape(world_size)

    graph = nx.DiGraph()
    k = grid_nodes.shape[0]*grid_nodes.shape[1]

    SxT_space = []
    for t in range(H):
        for i in range(np.prod(world_size)):
            SxT_space.append((t, i))

    graph.add_nodes_from(SxT_space, weight=0)

    for node in graph.nodes:
        graph.nodes[node]['weight'] = np.random.rand(1)[0]/100

    for t in range(H-1):
        # action 1: go right
        l = [(t, s) for s in grid_nodes[:, :-1].reshape(-1)]
        r = [(t+1, s) for s in grid_nodes[:, 1:].reshape(-1)]
        graph.add_edges_from(zip(l, r), action=1)

    for t in range(H-1):
        # action 1: go up
        l = [(t, s) for s in grid_nodes[:-1, :].reshape(-1)]
        r = [(t+1, s) for s in grid_nodes[1:, :].reshape(-1)]
        graph.add_edges_from(zip(l, r), action=2)

    for t in range(H-1):
        # action 1: go left
        l = [(t, s) for s in grid_nodes[:, 1:].reshape(-1)]
        r = [(t+1, s) for s in grid_nodes[:, :-1].reshape(-1)]
        graph.add_edges_from(zip(l, r), action=3)

    for t in range(H-1):
        # action 1: go down
        l = [(t, s) for s in grid_nodes[1:, :].reshape(-1)]
        r = [(t+1, s) for s in grid_nodes[:-1, :].reshape(-1)]
        graph.add_edges_from(zip(l, r), action=4)

    for t in range(H-1):
        # action 1: stay
        l = [(t, s) for s in grid_nodes[:, :].reshape(-1)]
        r = [(t+1, s) for s in grid_nodes[:, :].reshape(-1)]
        graph.add_edges_from(zip(l, r), action=0)

    return graph


def grid_world_graph(world_size):
    """Create a graph that represents a grid world.
    In the grid world there are four actions, (1, 2, 3, 4), which correspond
    to going (up, right, down, left) in the x-y plane. The states are
    ordered so that `np.arange(np.prod(world_size)).reshape(world_size)`
    corresponds to a matrix where increasing the row index corresponds to the
    x direction in the graph, and increasing y index corresponds to the y
    direction.
    Parameters
    ----------
    world_size: tuple
        The size of the grid world (rows, columns)
    Returns
    -------
    graph: nx.DiGraph()
        The directed graph representing the grid world.
    """
    nodes = np.arange(np.prod(world_size))
    grid_nodes = nodes.reshape(world_size)

    graph = nx.DiGraph()
    k = grid_nodes.shape[0]*grid_nodes.shape[1]

    graph.add_nodes_from([i for i in range(k)], weight=0)
    for node in graph.nodes:
        graph.nodes[node]['weight'] = np.random.rand(1)[0]/100

    # action 1: go right
    graph.add_edges_from(zip(grid_nodes[:, :-1].reshape(-1),
                             grid_nodes[:, 1:].reshape(-1)),
                         action=1)

    # action 2: go up
    graph.add_edges_from(zip(grid_nodes[:-1, :].reshape(-1),
                             grid_nodes[1:, :].reshape(-1)),
                         action=2)

    # action 3: go left
    graph.add_edges_from(zip(grid_nodes[:, 1:].reshape(-1),
                             grid_nodes[:, :-1].reshape(-1)),
                         action=3)

    # action 4: go down
    graph.add_edges_from(zip(grid_nodes[1:, :].reshape(-1),
                             grid_nodes[:-1, :].reshape(-1)),
                         action=4)

    # action 5: stay
    graph.add_edges_from(zip(grid_nodes[:, :].reshape(-1),
                             grid_nodes[:, :].reshape(-1)),
                         action=0)

    return graph


def room_grid_world_graph(world_size):
    """Create a graph that represents a grid world.
    In the grid world there are four actions, (1, 2, 3, 4), which correspond
    to going (up, right, down, left) in the x-y plane. The states are
    ordered so that `np.arange(np.prod(world_size)).reshape(world_size)`
    corresponds to a matrix where increasing the row index corresponds to the
    x direction in the graph, and increasing y index corresponds to the y
    direction.
    Parameters
    ----------
    world_size: tuple
        The size of the grid world (rows, columns)
    Returns
    -------
    graph: nx.DiGraph()
        The directed graph representing the grid world.
    """
    nodes = np.arange(np.prod(world_size))
    grid_nodes = nodes.reshape(world_size)

    graph = nx.DiGraph()
    k = grid_nodes.shape[0]*grid_nodes.shape[1]

    graph.add_nodes_from([i for i in range(k)], weight=0)
    for node in graph.nodes:
        graph.nodes[node]['weight'] = np.random.rand(1)[0]/100

    # action 1: go right
    a = grid_nodes[:, :-1].copy()
    a[0:2, 3:10] = np.zeros_like(a[0:2, 3:10])
    a[4:7, 3:10] = np.zeros_like(a[4:7, 3:10])
    b = grid_nodes[:, 1:].copy()
    b[0:2, 3:10] = np.zeros_like(b[0:2, 3:10])
    b[4:7, 3:10] = np.zeros_like(b[4:7, 3:10])
    graph.add_edges_from(zip(a.reshape(-1), b.reshape(-1)), action=1)

    # action 2: go up
    c = grid_nodes[:-1, :].copy()
    c[0:2, 4:10] = np.zeros_like(c[0:2, 4:10])
    c[3:7, 4:10] = np.zeros_like(c[3:7, 4:10])
    d = grid_nodes[1:, :].copy()
    d[0:2, 4:10] = np.zeros_like(d[0:2, 4:10])
    d[3:7, 4:10] = np.zeros_like(d[3:7, 4:10])
    graph.add_edges_from(zip(c.reshape(-1), d.reshape(-1)), action=2)

    # action 3: go left
    graph.add_edges_from(zip(b.reshape(-1), a.reshape(-1)), action=3)

    # action 4: go down
    graph.add_edges_from(zip(d.reshape(-1), c.reshape(-1)), action=4)

    # action 5: stay
    graph.add_edges_from(zip(grid_nodes[:, :].reshape(-1),
                             grid_nodes[:, :].reshape(-1)),
                         action=0)

    return graph


def state_reward_graph(world_size):
    nodes = np.arange(np.prod(world_size))
    grid_nodes = nodes.reshape(world_size)

    graph = nx.DiGraph()
    k = grid_nodes.shape[0]*grid_nodes.shape[1]

    graph.add_nodes_from([i for i in range(k)], weight=0)
    for node in graph.nodes:
        graph.nodes[node]['weight'] = np.random.rand(1)[0]/100
    return graph


def room_diag_reward_graph(world_size):
    """Create a graph that represents a grid world.
    In the grid world there are four actions, (1, 2, 3, 4), which correspond
    to going (up, right, down, left) in the x-y plane. The states are
    ordered so that `np.arange(np.prod(world_size)).reshape(world_size)`
    corresponds to a matrix where increasing the row index corresponds to the
    x direction in the graph, and increasing y index corresponds to the y
    direction.
    Parameters
    ----------
    world_size: tuple
        The size of the grid world (rows, columns)
    Returns
    -------
    graph: nx.DiGraph()
        The directed graph representing the grid world.
    """
    nodes = np.arange(np.prod(world_size))
    grid_nodes = nodes.reshape(world_size)

    graph = nx.DiGraph()

    # action 1: go right
    a = grid_nodes[:, :-1].copy()
    a[0:2, 3:10] = np.zeros_like(a[0:2, 3:10])
    a[4:7, 3:10] = np.zeros_like(a[4:7, 3:10])
    b = grid_nodes[:, 1:].copy()
    b[0:2, 3:10] = np.zeros_like(b[0:2, 3:10])
    b[4:7, 3:10] = np.zeros_like(b[4:7, 3:10])
    graph.add_edges_from(zip(a.reshape(-1), b.reshape(-1)), action=1)

    # action 2: go up
    c = grid_nodes[:-1, :].copy()
    c[0:2, 4:10] = np.zeros_like(c[0:2, 4:10])
    c[3:7, 4:10] = np.zeros_like(c[3:7, 4:10])
    d = grid_nodes[1:, :].copy()
    d[0:2, 4:10] = np.zeros_like(d[0:2, 4:10])
    d[3:7, 4:10] = np.zeros_like(d[3:7, 4:10])

    # action 2: go down
    graph.add_edges_from(zip(c.reshape(-1), d.reshape(-1)), action=2)

    # left up
    e = grid_nodes[:-1, :-1].copy()
    e[0:2, 4:10] = np.zeros_like(e[0:2, 4:10])
    e[4:6, 3:10] = np.zeros_like(e[4:6, 3:10])
    e[3, 3:9] = np.zeros_like(e[3, 3:9])
    e[0, 3] = np.zeros_like(e[0, 3])
    f = grid_nodes[1:, 1:].copy()
    f[0:2, 4:10] = np.zeros_like(f[0:2, 4:10])
    f[4:6, 3:10] = np.zeros_like(f[4:6, 3:10])
    f[3, 3:9] = np.zeros_like(f[3, 3:9])
    f[0, 3] = np.zeros_like(f[0, 3])
    graph.add_edges_from(zip(e.reshape(-1), f.reshape(-1)), action=5)

    return graph


def diag_reward_graph(world_size):
    """Create a graph that represents a grid world.
    In the grid world there are four actions, (1, 2, 3, 4), which correspond
    to going (up, right, down, left) in the x-y plane. The states are
    ordered so that `np.arange(np.prod(world_size)).reshape(world_size)`
    corresponds to a matrix where increasing the row index corresponds to the
    x direction in the graph, and increasing y index corresponds to the y
    direction.
    Parameters
    ----------
    world_size: tuple
        The size of the grid world (rows, columns)
    Returns
    -------
    graph: nx.DiGraph()
        The directed graph representing the grid world.
    """
    nodes = np.arange(np.prod(world_size))
    grid_nodes = nodes.reshape(world_size)

    graph = nx.DiGraph()

    # action 1: go right
    graph.add_edges_from(zip(grid_nodes[:, :-1].reshape(-1),
                             grid_nodes[:, 1:].reshape(-1)),
                         action=1)

    # action 2: go down
    graph.add_edges_from(zip(grid_nodes[:-1, :].reshape(-1),
                             grid_nodes[1:, :].reshape(-1)),
                         action=2)

    # # action 3: go left
    # graph.add_edges_from(zip(grid_nodes[:, 1:].reshape(-1),
    #                          grid_nodes[:, :-1].reshape(-1)),
    #                      action=3)

    # # action 4: go up
    # graph.add_edges_from(zip(grid_nodes[1:, :].reshape(-1),
    #                          grid_nodes[:-1, :].reshape(-1)),
    #                      action=4)

    # # left up
    graph.add_edges_from(zip(grid_nodes[:-1, :-1].reshape(-1),
                             grid_nodes[1:, 1:].reshape(-1)),
                         action=5)

    # graph.add_edges_from(zip(grid_nodes[:-1, 1:].reshape(-1),
    #                          grid_nodes[1:, :-1].reshape(-1)),
    #                      action=6)
    # graph.add_edges_from(zip(grid_nodes[1:, :-1].reshape(-1),
    #                          grid_nodes[:-1, 1:].reshape(-1)),
    #                      action=7)
    # graph.add_edges_from(zip(grid_nodes[1:, 1:].reshape(-1),
    #                          grid_nodes[:-1, :-1].reshape(-1)),
    #                      action=8)

    return graph


def horizon_invert_diag_reward_graph(world_size, H):
    """Create a graph that represents a grid world.
    In the grid world there are four actions, (1, 2, 3, 4), which correspond
    to going (up, right, down, left) in the x-y plane. The states are
    ordered so that `np.arange(np.prod(world_size)).reshape(world_size)`
    corresponds to a matrix where increasing the row index corresponds to the
    x direction in the graph, and increasing y index corresponds to the y
    direction.
    Parameters
    ----------
    world_size: tuple
        The size of the grid world (rows, columns)
    Returns
    -------
    graph: nx.DiGraph()
        The directed graph representing the grid world.
    """
    nodes = np.arange(np.prod(world_size))
    grid_nodes = nodes.reshape(world_size)

    graph = nx.DiGraph()

    # # action 1: go right
    # graph.add_edges_from(zip(grid_nodes[:, :-1].reshape(-1),
    #                          grid_nodes[:, 1:].reshape(-1)),
    #                      action=1)

    # # action 2: go up
    # graph.add_edges_from(zip(grid_nodes[:-1, :].reshape(-1),
    #                          grid_nodes[1:, :].reshape(-1)),
    #                      action=2)

    for t in range(H-1):
        # action 3: go left
        l = [(t, s) for s in grid_nodes[:, 1:].reshape(-1)]
        r = [(t+1, s) for s in grid_nodes[:, :-1].reshape(-1)]
        graph.add_edges_from(zip(l, r), action=3)

    for t in range(H-1):
        # action 1: go down
        l = [(t, s) for s in grid_nodes[:-1, :].reshape(-1)]
        r = [(t+1, s) for s in grid_nodes[1:, :].reshape(-1)]
        graph.add_edges_from(zip(l, r), action=4)

    # # right up
    # graph.add_edges_from(zip(grid_nodes[:-1, :-1].reshape(-1),
    #                          grid_nodes[1:, 1:].reshape(-1)),
    #                      action=5)

    # graph.add_edges_from(zip(grid_nodes[:-1, 1:].reshape(-1),
    #                          grid_nodes[1:, :-1].reshape(-1)),
    #                      action=6)
    # graph.add_edges_from(zip(grid_nodes[1:, :-1].reshape(-1),
    #                          grid_nodes[:-1, 1:].reshape(-1)),
    #                      action=7)

    # # left down
    for t in range(H-1):
        # action 1: left down
        l = [(t, s) for s in grid_nodes[1:, 1:].reshape(-1)]
        r = [(t+1, s) for s in grid_nodes[:-1, :-1].reshape(-1)]
        graph.add_edges_from(zip(l, r), action=8)
    # graph.add_edges_from(zip(grid_nodes[1:, 1:].reshape(-1),
    #                          grid_nodes[:-1, :-1].reshape(-1)),
    #                      action=8)

    for t in range(H-1):
        # action 1: stay
        l = [(t, s) for s in grid_nodes[:, :].reshape(-1)]
        r = [(t+1, s) for s in grid_nodes[:, :].reshape(-1)]
        graph.add_edges_from(zip(l, r), action=5)

    return graph


def invert_diag_reward_graph(world_size):
    """Create a graph that represents a grid world.
    In the grid world there are four actions, (1, 2, 3, 4), which correspond
    to going (up, right, down, left) in the x-y plane. The states are
    ordered so that `np.arange(np.prod(world_size)).reshape(world_size)`
    corresponds to a matrix where increasing the row index corresponds to the
    x direction in the graph, and increasing y index corresponds to the y
    direction.
    Parameters
    ----------
    world_size: tuple
        The size of the grid world (rows, columns)
    Returns
    -------
    graph: nx.DiGraph()
        The directed graph representing the grid world.
    """
    nodes = np.arange(np.prod(world_size))
    grid_nodes = nodes.reshape(world_size)

    graph = nx.DiGraph()

    # # action 1: go right
    # graph.add_edges_from(zip(grid_nodes[:, :-1].reshape(-1),
    #                          grid_nodes[:, 1:].reshape(-1)),
    #                      action=1)

    # # action 2: go up
    # graph.add_edges_from(zip(grid_nodes[:-1, :].reshape(-1),
    #                          grid_nodes[1:, :].reshape(-1)),
    #                      action=2)

    # action 3: go left
    graph.add_edges_from(zip(grid_nodes[:, 1:].reshape(-1),
                             grid_nodes[:, :-1].reshape(-1)),
                         action=3)

    # action 4: go down
    graph.add_edges_from(zip(grid_nodes[1:, :].reshape(-1),
                             grid_nodes[:-1, :].reshape(-1)),
                         action=4)

    # # right up
    # graph.add_edges_from(zip(grid_nodes[:-1, :-1].reshape(-1),
    #                          grid_nodes[1:, 1:].reshape(-1)),
    #                      action=5)

    # graph.add_edges_from(zip(grid_nodes[:-1, 1:].reshape(-1),
    #                          grid_nodes[1:, :-1].reshape(-1)),
    #                      action=6)
    # graph.add_edges_from(zip(grid_nodes[1:, :-1].reshape(-1),
    #                          grid_nodes[:-1, 1:].reshape(-1)),
    #                      action=7)

    # # left down
    graph.add_edges_from(zip(grid_nodes[1:, 1:].reshape(-1),
                             grid_nodes[:-1, :-1].reshape(-1)),
                         action=8)

    return graph


def nodes_to_states(nodes, world_shape, step_size):
    """Convert node numbers to physical states.
    Parameters
    ----------
    nodes: np.array
        Node indices of the grid world
    world_shape: tuple
        The size of the grid_world
    step_size: np.array
        The step size of the grid world
    Returns
    -------
    states: np.array
        The states in physical coordinates
    """
    nodes = torch.as_tensor(nodes)
    step_size = torch.as_tensor(step_size)
    return torch.vstack(((nodes // world_shape["y"]),
                         (nodes % world_shape["y"]))).T * step_size


def grid(world_shape, step_size, start_loc):
    """
    Creates grids of coordinates and indices of state space
    Parameters
    ----------
    world_shape: tuple
        Size of the grid world (rows, columns)
    step_size: tuple
        Phyiscal step size in the grid world
    Returns
    -------
    states_ind: np.array
        (n*m) x 2 array containing the indices of the states
    states_coord: np.array
        (n*m) x 2 array containing the coordinates of the states
    """
    nodes = torch.arange(0, world_shape["x"] * world_shape["y"])
    return nodes_to_states(nodes, world_shape, step_size) + start_loc


class GridWorld():
    def __init__(self, env_params, common_params, visu_params, env_file_path):
        self.env_params = env_params
        self.common_params = common_params
        self.gridV = grid(
            env_params["shape"], env_params["step_size"], env_params["start"])

        if self.env_params["domains"]=="two_room":
            self.rew_graph = room_diag_reward_graph(
                (self.env_params['shape']['x'], self.env_params['shape']['y']))
            # self.horizon_transition_graph = horizon_grid_world_graph(
            #     (self.env_params['shape']['x'], self.env_params['shape']['y']), env_params["horizon"])
            self.transition_graph = room_grid_world_graph(
                (self.env_params['shape']['x'], self.env_params['shape']['y']))
        else: 
            self.rew_graph = diag_reward_graph(
            (self.env_params['shape']['x'], self.env_params['shape']['y']))
        # self.horizon_transition_graph = horizon_grid_world_graph(
        #     (self.env_params['shape']['x'], self.env_params['shape']['y']), env_params["horizon"])
            self.transition_graph = grid_world_graph(
                (self.env_params['shape']['x'], self.env_params['shape']['y']))

        self.done = False
        self.env_size = self.gridV.shape[0]
        self.horizon = env_params["horizon"]
        self.node_size = env_params["shape"]['x'] * env_params["shape"]['y']
        self.action_dim = 5
        if env_params["node_weight"] == "steiner_covering":
            # self.Dmin = [13, 6, 10, 4, 7, 15] # minimum number of items to be collected
            # self.Gmax = [28, 24, 23, 26, 28, 24] # maximum number of items in the environment
            self.Dmin = [3, 2, 3, 4, 6, 3] # minimum number of items to be collected
            self.Gmax = [18, 14, 13, 16, 18, 14] # maximum number of items in the environment
            if self.env_params["generate"]:
                self.generate_Gi_Di()
                a_file = open(env_file_path, "wb")
                pickle.dump(self.items_loc, a_file)
                a_file.close()
        elif env_params["node_weight"] == "entropy":
            if self.env_params["generate"]:
                self.Fx_noise = env_params["Fx_noise"]
                self.Fx_covar_module = ScaleKernel(base_kernel=MaternKernel(nu=2.5),)
                self.Fx_lengthscale = env_params["Fx_lengthscale"]
                self.generate_multi_distribution()
                a_file = open(env_file_path, "wb")
                pickle.dump(self.cov, a_file)
                a_file.close()
        elif env_params["node_weight"] == "GP":
            if self.env_params["generate"]:
                self.Fx_noise = env_params["Fx_noise"]
                self.Fx_covar_module = ScaleKernel(base_kernel=MaternKernel(nu=2.5),)
                self.Fx_lengthscale = env_params["Fx_lengthscale"]
                density = self.__true_density_sampling()
                nx, ny = (self.env_params['shape']['x'], self.env_params['shape']['y'])
                self.weight = {}
                for n in range(self.node_size):
                    self.weight[n] = density[n].item()
                a_file = open(env_file_path, "wb")
                pickle.dump(self.weight, a_file)
                a_file.close()
            self.gen_coverage_map()
        elif env_params["node_weight"] == "gorilla":
            density = GridFunction((30, 30), get_gorillas_density())
            sx = env_params["shape"]["x"]
            density.upsample(current=True, factors=(sx / 100, sx / 100))
            nx, ny = (self.env_params['shape']['x'], self.env_params['shape']['y'])
            self.weight = {}
            for n in range(self.node_size):
                x = n % nx
                y = int(n/nx)
                self.weight[n] = density.current_function[x,y]
            self.gen_coverage_map()
        else:
            self.get_groundtruth_weights()
            self.gen_coverage_map()
            if self.env_params["disc_size"] == "large":
                self.modify_disc_size()
        # if env_params["generate"] == True:
        #     self.__Fx = self.true_density_sampling()
        #     a = self.__Fx
        #     self.env_data["Fx"] = self.__Fx

        # if common_params["grad"] == "sym":
        #     self.X = symarray(
        #         'A', (self.env_size, self.env_params["horizon"]))
        #     self.F = self.coverage_function(self.X)
        #     # F = X[1, 2] + X[3, 4]*10 + X[3, 3]*X[5, 3]
        #     self.DF = derive_by_array(self.F, self.X)

    def step(self, h, action):
        if self.env_params["stochasticity"]>0:
            len_rand = int(self.env_params["stochasticity"]*action.shape[0])
            rand_actions = torch.randint(0,5,(len_rand,))
            rand_loc = torch.randint(0,action.shape[0],(len_rand,))
            action[rand_loc] = rand_actions
            self.state = self.Hori_ActionTransitionMatrix[:, self.state, action].argmax(0)
        else:
            self.state = self.Hori_ActionTransitionMatrix[:, self.state, action].argmax(0)
        return self.state

    def initialize(self):
        # self.state = 0*self.dPi_init.multinomial(
        #     self.common_params["batch_size"], replacement=True)
        self.state = 34*torch.ones(self.common_params["batch_size"], dtype=int)
        return self.state

    def dist(self, x, y):
        ret = np.reciprocal(1+np.exp(-x*y))  # + np.abs(x)*y  # + np.abs(x)*y
        return ret

    def get_groundtruth_weights(self):
        self.weight = {}
        nx, ny = (self.env_params['shape']['x'], self.env_params['shape']['y'])
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        xv, yv = np.meshgrid(x, y)
        val = self.dist(xv, yv)
        for n in range(self.node_size):
            x = n % nx
            y = int(n/nx)
            if self.env_params["node_weight"] == "bimodal":
                self.weight[n] = val[x, y]
            elif self.env_params["node_weight"] == "constant":
                self.weight[n] = 1
            elif self.env_params["node_weight"] == "linear":
                self.weight[n] = n

    def get_J_pi_policy(self, pathMatrix, dPi_init, paths_reward, pi_h_s_a):
        # X_h_s_a = torch.ones(self.horizon-1, self.node_size,
        #                      self.action_dim, requires_grad=True)
        P_s_old = dPi_init.clone()
        ret = 0
        temp_pi_h_s_a = torch.concat([(torch.ones_like(pi_h_s_a[:, :, 0]) - torch.sum(
            pi_h_s_a[:, :, 1:], 2)).reshape(self.horizon-1, -1, 1), pi_h_s_a[:, :, 1:]], 2)
        for traj_idx in range(pathMatrix.shape[1]):
            # print(traj_idx)
            P_s_old = dPi_init.clone()
            for h_idx in range(self.horizon-1):
                st_idx = h_idx*self.node_size*self.action_dim
                ed_idx = (h_idx+1)*self.node_size*self.action_dim
                hot_vec = pathMatrix[:, traj_idx][st_idx:ed_idx].reshape(
                    self.node_size, self.action_dim)
                P_s_hot = torch.mul(hot_vec.transpose(
                    0, 1), P_s_old).transpose(0, 1)
                # P_sto_sfrom = torch.sum(
                #     torch.mul(self.PA, pi_s_a_h[:, :, h_idx]), 2)
                P_sto_sfrom_act = torch.mul(
                    self.Hori_ActionTransitionMatrix[:, :, :, h_idx], temp_pi_h_s_a[h_idx, :, :])
                # torch.matmul(P_sto_sfrom_act,hot_vec.transpose(0,1))
                P_s_next = torch.sum(
                    torch.mul(P_sto_sfrom_act, P_s_hot), (1, 2))
                P_s_old = P_s_next.clone()
            hot_vec = pathMatrix[:, traj_idx][-self.node_size:]
            alpha = torch.matmul(hot_vec, P_s_old)
            ret += alpha*paths_reward[traj_idx]
            # print(ret)
        return ret

    def get_J_pi_X_py(self, pathMatrix, dPi_init, paths_reward, X_h_s_a):
        # X_h_s_a = torch.ones(self.horizon-1, self.node_size,
        #                      self.action_dim, requires_grad=True)
        pi_h_s_a = torch.divide(X_h_s_a, torch.sum(X_h_s_a, 2)[:, :, None])
        pi_h_s_a[pi_h_s_a != pi_h_s_a] = 0
        P_s_old = dPi_init.clone()
        ret = 0
        for traj_idx in range(pathMatrix.shape[1]):
            # print(traj_idx)
            P_s_old = dPi_init.clone()
            for h_idx in range(self.horizon-1):
                st_idx = h_idx*self.node_size*self.action_dim
                ed_idx = (h_idx+1)*self.node_size*self.action_dim
                hot_vec = pathMatrix[:, traj_idx][st_idx:ed_idx].reshape(
                    self.node_size, self.action_dim)
                P_s_hot = torch.mul(hot_vec.transpose(
                    0, 1), P_s_old).transpose(0, 1)
                # P_sto_sfrom = torch.sum(
                #     torch.mul(self.PA, pi_s_a_h[:, :, h_idx]), 2)
                P_sto_sfrom_act = torch.mul(
                    self.Hori_ActionTransitionMatrix[:, :, :, h_idx], pi_h_s_a[h_idx, :, :])
                # torch.matmul(P_sto_sfrom_act,hot_vec.transpose(0,1))
                P_s_next = torch.sum(
                    torch.mul(P_sto_sfrom_act, P_s_hot), (1, 2))
                P_s_old = P_s_next.clone()
            hot_vec = pathMatrix[:, traj_idx][-self.node_size:]
            alpha = torch.matmul(hot_vec, P_s_old)
            ret += alpha*paths_reward[traj_idx]
            # print(ret)
        return ret

    def get_J_pi_X(self, pathMatrix, dPi_init, paths_reward):
        pi_h_s_a = np.divide(self.X_h_s_a, np.sum(self.X_h_s_a, 2)[:, :, None])
        P_s_old = dPi_init.clone().numpy()
        pathMatrix = pathMatrix.clone().numpy()
        ret = 0
        for traj_idx in range(pathMatrix.shape[1]):
            # print(traj_idx)
            P_s_old = dPi_init.clone().numpy()
            for h_idx in range(self.horizon-1):
                st_idx = h_idx*self.node_size*self.action_dim
                ed_idx = (h_idx+1)*self.node_size*self.action_dim
                hot_vec = pathMatrix[:, traj_idx][st_idx:ed_idx].reshape(
                    self.node_size, self.action_dim)
                P_s_hot = np.multiply(hot_vec.transpose(
                    0, 1), P_s_old[:, None]).transpose(0, 1)
                # P_sto_sfrom = torch.sum(
                #     torch.mul(self.PA, pi_s_a_h[:, :, h_idx]), 2)
                P_sto_sfrom_act = np.multiply(
                    self.Hori_ActionTransitionMatrix[:, :, :, 0].numpy(), pi_h_s_a[h_idx, :, :])
                # torch.matmul(P_sto_sfrom_act,hot_vec.transpose(0,1))
                P_s_next = np.sum(np.multiply(
                    P_sto_sfrom_act, P_s_hot), (1, 2))
                P_s_old = P_s_next.copy()
            hot_vec = pathMatrix[:, traj_idx][-self.node_size:]
            alpha = np.matmul(hot_vec, P_s_old)
            ret += alpha*paths_reward[traj_idx]
            # print(ret)
        self.J_pi_X = ret
        return ret

    def get_D_J_pi_X(self):
        self.D_J_pi_X = derive_by_array(self.J_pi_X, self.X_h_s_a)

    def get_FW_obj(self, dPi):
        subs_dict = dict(zip(self.X_h_s_a.ravel(), dPi.numpy().ravel()))
        return self.J_pi_X.subs(subs_dict)

    def get_FW_grad(self, dPi):
        subs_dict = dict(zip(self.X_h_s_a.ravel(), dPi.numpy().ravel()))
        return self.D_J_pi_X.subs(subs_dict)
    # def step(self, action1, action2):
    #     action1 = action1.reshape((self.d, 1))
    #     action2 = action2.reshape((self.d, 1))

    #     self.state = self.A*self.state + self.B1*action1 - self.B2*action2
    #     r0 = -1*action1*action1 + 1*action2*action2 + self.state * \
    #         self.state  # action1.transpose(0,1)*self.W12*action2.reshape(1)
    #     self.reward = np.array((r0, r0))
    #     info = {}
    #     return self.state, self.reward[0], self.reward[1], False, info

    def traj_weight(self, n, traj, SxT_space, weights_mat):
        rem_states = copy.copy(SxT_space)
        random.shuffle(rem_states)
        w_traj = traj + rem_states
        # print("env-size", self.env_size)
        # what heppens if the same state is visited twice, how to take that into account?
        # weights_mat.append({})
        for i, ele in enumerate(w_traj):
            weights_mat[n][ele] = self.submodular_return(
                w_traj[0:i+1]) - self.submodular_return(w_traj[0:i])

    def get_weights_parallelized(self, traj):
        """ returns a vector m, w_i.
        for any trajectory we can pick any permutation, but let say there is only one rule for now.
        Based on this permutation and trajectory we can get weights vector as per this traj
        now every trajectory with have a "m" vector denoting it contribution to the weight vector.
        Now the inner product (between weights(hypercube corners) and m) is the actual subgradient

        Args:
            traj (_type_): tuple of time and state (t,s) of length = horizon

        Returns:
            _type_: _description_
        """

        SxT_space = []
        for t in range(self.env_params["horizon"]):
            for i in range(self.env_size):
                if (t, i) not in traj:
                    SxT_space.append((t, i))
        # rem_states = set(SxT_space) - set(traj)
        manager = multiprocessing.Manager()
        jobs = []
        weights_mat = manager.dict()
        for n in range(10):
            weights_mat[n] = manager.dict()
            p = multiprocessing.Process(
                target=self.traj_weight, args=(n, traj, SxT_space, weights_mat))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        weights = {}
        for t in range(self.env_params["horizon"]):
            for s in range(self.env_size):
                weights[(t, s)] = np.average([traj_sg[(t, s)]
                                              for traj_sg in weights_mat.values()])
        # get it by finding an expected path(sample path) using the edges on the remaining states. (what if a state is visited twice or there is no way to reach one of the reamaining states without revisiting a few)
        # append it to the weights ....so that there is weight corresponding to every trajectory.
        # (some how permutation needs to take time into account or remove the states which are visited twice or multiple times, and compute based on the newly visited states only)
        return weights

    def get_weights(self, traj):
        """ returns a vector m, w_i.
        for any trajectory we can pick any permutation, but let say there is only one rule for now.
        Based on this permutation and trajectory we can get weights vector as per this traj
        now every trajectory with have a "m" vector denoting it contribution to the weight vector.
        Now the inner product (between weights(hypercube corners) and m) is the actual subgradient

        Args:
            traj (_type_): tuple of time and state (t,s) of length = horizon

        Returns:
            _type_: _description_
        """
        weights_mat = []

        SxT_space = []
        for t in range(self.env_params["horizon"]):
            for i in range(self.env_size):
                if (t, i) not in traj:
                    SxT_space.append((t, i))
        # rem_states = set(SxT_space) - set(traj)
        for n in range(1):
            rem_states = copy.copy(SxT_space)
            # random.shuffle(rem_states)
            w_traj = traj + rem_states
            # what heppens if the same state is visited twice, how to take that into account?
            weights_mat.append({})
            for i, ele in enumerate(w_traj):
                weights_mat[-1][ele] = self.submodular_return(
                    w_traj[0:i+1]) - self.submodular_return(w_traj[0:i])
        weights = {}
        for t in range(self.env_params["horizon"]):
            for s in range(self.env_size):
                weights[(t, s)] = np.average([traj_sg[(t, s)]
                                              for traj_sg in weights_mat])
        # get it by finding an expected path(sample path) using the edges on the remaining states. (what if a state is visited twice or there is no way to reach one of the reamaining states without revisiting a few)
        # append it to the weights ....so that there is weight corresponding to every trajectory.
        # (some how permutation needs to take time into account or remove the states which are visited twice or multiple times, and compute based on the newly visited states only)
        return weights

    def get_uniform_weights(self, traj):
        """ returns a vector m, w_i.
        for any trajectory we can pick any permutation, but let say there is only one rule for now.
        Based on this permutation and trajectory we can get weights vector as per this traj
        now every trajectory with have a "m" vector denoting it contribution to the weight vector.
        Now the inner product (between weights(hypercube corners) and m) is the actual subgradient

        Args:
            traj (_type_): tuple of time and state (t,s) of length = horizon

        Returns:
            _type_: _description_
        """
        weights_mat = []

        SxT_space = []
        traj_states = [s for (i, s) in traj]

        permute = copy(traj_states)
        if self.common_params["subgrad"] == "greedy":
            # identify the neighbours for the terminal state
            neighbor_exists = True
            while neighbor_exists:  # if it is -1 then there are no new neighbours to go
                terminal_state = permute[-1]
                neighbors = list(
                    self.transition_graph.neighbors(terminal_state))
                neighbor_exists = False
                marginal_gain_max = -1  # initialize to zero and then pick the agent with maximum gain
                for neighbor_state in neighbors:
                    if neighbor_state not in permute:
                        marginal_gain = self.submodular_return_states(
                            permute + [neighbor_state]) - self.submodular_return_states(permute)
                        if marginal_gain > marginal_gain_max:
                            curr_expand_state = neighbor_state
                            marginal_gain_max = marginal_gain
                            neighbor_exists = True
                if neighbor_exists:
                    permute.append(curr_expand_state)
            # and put the greedy element at the end of the trajectory
            s_rem = []
            for s in range(self.env_size):
                if s not in permute:
                    s_rem.append(s)

            random.shuffle(s_rem)
            rem_states = [(0, s) for s in (permute[len(traj_states):] + s_rem)]

        else:
            s_rem = []
            for s in range(self.env_size):
                if s not in traj_states:
                    s_rem.append(s)

            rem_states = [(0, s) for s in s_rem]
            random.shuffle(rem_states)

        w_traj = traj + rem_states
        # what heppens if the same state is visited twice, how to take that into account?
        weights_mat.append({})
        for i, ele in enumerate(w_traj):
            weights_mat[-1][ele] = self.submodular_return(
                w_traj[0:i+1]) - self.submodular_return(w_traj[0:i])

        weights = {}
        for t in range(self.env_params["horizon"]):
            for s in range(self.env_size):
                if (t, s) in traj:
                    weights[(t, s)] = weights_mat[-1][(t, s)]
                elif (0, s) in rem_states:
                    weights[(t, s)] = weights_mat[-1][(0, s)] / \
                        self.env_params["horizon"]
                else:
                    weights[(t, s)] = 0
                    # np.average([traj_sg[(t, s)]
                    #             for traj_sg in weights_mat])
        # get it by finding an expected path(sample path) using the edges on the remaining states. (what if a state is visited twice or there is no way to reach one of the reamaining states without revisiting a few)
        # append it to the weights ....so that there is weight corresponding to every trajectory.
        # (some how permutation needs to take time into account or remove the states which are visited twice or multiple times, and compute based on the newly visited states only)
        return weights

    def random_dpi(self, pathMatrix, pathMatrix_s):
        alpha = torch.rand(pathMatrix.shape[1])
        alpha = alpha/torch.sum(alpha)
        # for dpi (h,s)
        # dpi = torch.sum(torch.mul(pathMatrix_s, alpha),1).reshape(self.horizon, -1).transpose(0, 1)
        # for pathmatrix h,s,a
        dpi = torch.sum(torch.mul(pathMatrix, alpha), 1)
        dpi_action = dpi[:-self.node_size].reshape(
            self.horizon-1, self.node_size, self.action_dim)
        dpi = torch.cat([torch.sum(dpi_action, 2),
                        dpi[-self.node_size:].reshape(1, -1)], 0).transpose(0, 1)
        for i in range(dpi.shape[1]-1):
            dpi[:, i+1] = dpi[:, i]
        # dpi_action = dpi_action*0
        # dpi_action[:, :, 0] = torch.ones_like(dpi_action[:, :, 0])
        return dpi, dpi_action

    def submodular_return(self, traj):
        cover_node = set()
        for t, node in traj:
            edges = self.rew_graph.edges(node)
            connected_nodes = [node] + [v for u, v in edges]
            cover_node = cover_node.union(set(connected_nodes))
        return len(cover_node)

    def weighted_traj_return(self, mat_state, type="M"):
        if self.env_params["node_weight"] == "constant" or self.env_params["node_weight"] == "gorilla" or self.env_params["node_weight"] == "GP" or self.env_params["node_weight"] == "bimodal":
            return self.ret_cell_coverage(mat_state, type)
        elif self.env_params["node_weight"] == "steiner_covering":
            return self.ret_stiener_covering(mat_state, type)
        elif self.env_params["node_weight"] == "entropy":
            return self.ret_entropy(mat_state, type)
    
    def generate_multi_distribution(self):
        n = 5
        self.Fx_X = (torch.rand(2*n) * 10).reshape(-1, 2)
        self.Fx_Y = torch.zeros(self.Fx_X.shape[0], 1)
        self.Fx_model = SingleTaskGP(
            self.Fx_X, self.Fx_Y, covar_module=self.Fx_covar_module
        )
        self.Fx_model.covar_module.base_kernel.lengthscale = self.Fx_lengthscale
        self.Fx_model.likelihood.noise = self.Fx_noise
        self.cov = self.Fx_model.posterior(self.gridV).mvn.covariance_matrix.detach()        

    def generate_Gi_Di(self):
        items = {"apple", "oranges", "watermelon", "banana", "pineapple", "strawberries"} # cherry, grapes, guvava, kiwi, lemon
        self.items_loc = dict.fromkeys(items) # set of locations which has apple
        # self.Gmax = [8, 10, 12, 16, 8, 14] 
        perm = torch.randperm(self.node_size)[:np.sum(self.Gmax)]
        cum_idx = 0
        for i, item in enumerate(items):
            # find the location to keep items
            self.items_loc[item] = set(perm[cum_idx : cum_idx + self.Gmax[i]].tolist()) # make a set
            cum_idx += self.Gmax[i]
        self.steiner_map = perm
        

    def ret_cell_coverage(self, mat_state, type="M"):
        traj = torch.vstack(mat_state)
        returns = []
        for i in range(traj.shape[1]):
            coverage_i = itemgetter(*traj[:, i].tolist())(self.coverage_map)
            cover_node = []
            for cover in coverage_i:
                cover_node += cover
            if type=="SRL":
                ret = np.sum([self.weight[n] for n in cover_node])
            else:
                ret = np.sum([self.weight[n] for n in set(cover_node)])
            returns.append(ret)
        return torch.from_numpy(np.hstack(returns))

    def ret_stiener_covering(self, mat_state, type):
        traj = torch.vstack(mat_state)
        returns = []
        self.steiner_map = []
        for key in self.items_loc:
            self.steiner_map+=list(self.items_loc[key])
        if type=="SRL":
            for i in range(traj.shape[1]):
                # unique, counts = np.unique(traj[:, i].numpy(), return_counts=True)
                list_Ele = np.intersect1d(traj[:, i].numpy(), self.steiner_map)
                Fs = 0
                for ele in list_Ele:
                    Fs += np.sum(traj[:, i].numpy()==ele)
                returns.append(Fs)
        else:
            for i in range(traj.shape[1]):
                Fs = 0
                for k, item in enumerate(self.items_loc):
                    item_coverage = self.items_loc[item].intersection(set(traj[:, i].tolist()))
                    num_items = len(item_coverage)
                    if num_items < self.Dmin[k]:
                        Fs+=num_items
                    else:
                        Fs+=self.Dmin[k]
                returns.append(Fs)
        return torch.from_numpy(np.hstack(returns))
    
    def ret_entropy(self, mat_state, type):
        # https://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/
        traj = torch.vstack(mat_state)
        returns = []
        # with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #     self.Fx_model.eval()
        #     for i in range(traj.shape[1]):
        #         Fs_ent = self.Fx_model.posterior(self.gridV[traj[:,i].tolist()]).mvn.entropy().detach()
        #         returns.append(Fs_ent)
        if type=="SRL":
            for i in range(traj.shape[1]):
                # 1/2*(log(2 pi sigma^2)) + (1+1.837877)/2.0 - (1+1.837877)/2.0 - log(\sigma^2)/2
                # Fs_ent = torch.sum(torch.log2(self.cov.diag()[traj[:,i]]+ 1e-6)/2.0 + (1+1.837877)/2.0)
                Fs_info = torch.sum(torch.log2(self.cov.diag()[traj[:,i]]+ 1e-6)/2.0 + 9.965784)
                returns.append(Fs_info)
        else:
            for i in range(traj.shape[1]):
                # 1/2*(log(2 pi sigma^2)) + 1/2, torch.log2(1e-3) = -9.965784
                # Fs_ent = torch.logdet(self.cov[traj[:,i]][:,traj[:,i]] + 1e-6*torch.eye(traj.shape[0]))/2.0 + (1+1.837877)*traj.shape[0]/2.0
                Fs_info = torch.logdet(self.cov[traj[:,i]][:,traj[:,i]] + 1e-6*torch.eye(traj.shape[0]))/2.0 + 9.965784*traj.shape[0] 
                returns.append(Fs_info)
        return torch.hstack(returns)

    def weighted_submodular_return(self, traj):
        cover_node = set()
        for t, node in traj:
            edges = self.rew_graph.edges(node)
            connected_nodes = [node] + [v for u, v in edges]
            cover_node = cover_node.union(set(connected_nodes))
        ret = np.sum([self.weight[n] for n in cover_node])
        return ret

    def batched_marginal_coverage(self, mat_state, traj_small):
        traj_large = torch.vstack(mat_state)
        returns = []
        for i in range(traj_large.shape[1]):
            idx_traj = []
            for h in range(traj_large.shape[0]):
                idx_traj.append([h, traj_large[h, i].item()])
            ret = self.marginal_coverage(idx_traj, [traj_small[0][i].item()])
            returns.append(ret)
        return torch.from_numpy(np.hstack(returns))

    def marginal_coverage(self, traj_large, traj_small):
        cover_node = set()
        for t, node in traj_large:
            edges = self.rew_graph.edges(node)
            connected_nodes = [node] + [v for u, v in edges]
            cover_node = cover_node.union(set(connected_nodes))
        sub_node = set()
        for node in traj_small:
            edges = self.rew_graph.edges(node)
            connected_nodes = [node] + [v for u, v in edges]
            sub_node = sub_node.union(set(connected_nodes))
        marginal_node = cover_node - sub_node
        ret = np.sum([self.weight[n] for n in marginal_node])
        return ret

    def gen_coverage_map(self):
        self.coverage_map = dict()
        for node in self.rew_graph.nodes:
            edges = self.rew_graph.edges(node)
            connected_nodes = [node] + [v for u, v in edges]
            self.coverage_map[node] = connected_nodes

    def modify_disc_size(self):
        ex_cov_map = dict()
        for node in self.coverage_map.keys():
            ex_cov = []
            for cell in self.coverage_map[node]:
                ex_cov += self.coverage_map[cell]
            ex_cov_map[node] = ex_cov
        self.coverage_map = ex_cov_map

    def submodular_return_states(self, state_traj):
        cover_node = set()
        for node in state_traj:
            edges = self.rew_graph.edges(node)
            connected_nodes = [node] + [v for u, v in edges]
            cover_node = cover_node.union(set(connected_nodes))
        return len(cover_node)

    def get_horizon_transition_matrix(self):
        # self.Hori_ActionTransitionMatrix = torch.zeros(
        #     self.node_size, self.node_size, self.action_dim, self.env_params["horizon"]-1)
        # for h in range(self.env_params["horizon"]-1):
        #     for s, s_dash, data in self.transition_graph.edges(data=True):
        #         self.Hori_ActionTransitionMatrix[s_dash,
        #                                          s, data['action'], h] = 1.0
        self.Hori_ActionTransitionMatrix = torch.zeros(
            self.node_size, self.node_size, self.action_dim)
        for s, s_dash, data in self.transition_graph.edges(data=True):
            self.Hori_ActionTransitionMatrix[s_dash,
                                             s, data['action']] = 1.0
        for s in range(self.node_size):
            for a in range(self.action_dim):
                if torch.sum(self.Hori_ActionTransitionMatrix[:, s, a])==0:
                    self.Hori_ActionTransitionMatrix[s, s, a] = 1.0

    def stationary_pi(self, dpi_init):
        val = []
        for s in range(self.node_size):
            val.append(self.weighted_submodular_return(
                [(h, s) for h in range(self.horizon)]))
        ret = torch.sum(torch.mul(dpi_init, torch.Tensor(val)))
        return ret

    def optimal_J_pi(self, pathMatrix, paths_reward, dPi_init):
        rew = -np.array(paths_reward)
        # A_ub = -np.eye(pathMatrix.shape[1])
        # b_ub = np.zeros(pathMatrix.shape[1])
        # dPi_init constraints
        A_dpi_init = torch.zeros(dPi_init.shape[0], pathMatrix.shape[0])
        A_struct = torch.zeros(
            dPi_init.shape[0], self.node_size*self.action_dim)
        for s_idx in range(dPi_init.shape[0]):
            A_dpi_init[s_idx, s_idx*self.action_dim:s_idx*self.action_dim +
                       self.action_dim] = torch.ones(self.action_dim).reshape(1, -1)
            A_struct[s_idx, s_idx*self.action_dim:s_idx*self.action_dim +
                     self.action_dim] = torch.ones(self.action_dim).reshape(1, -1)

        # transition costraints: state visitation polytope constraint
        A_trans_const = torch.zeros(
            (self.horizon-1)*self.node_size, pathMatrix.shape[0])
        for h_idx in range(self.horizon-1):
            st_idx = h_idx*self.node_size
            st_act_idx = h_idx*self.node_size*self.action_dim
            ed_idx = (h_idx+1)*self.node_size*self.action_dim
            A_trans_const[st_idx:st_idx+self.node_size, st_act_idx:st_act_idx+self.node_size *
                          self.action_dim] = self.Hori_ActionTransitionMatrix[:, :, :, h_idx].reshape(self.node_size, -1)
            if h_idx < self.horizon - 2:
                A_trans_const[st_idx:st_idx+self.node_size,
                              ed_idx:ed_idx+self.node_size*self.action_dim] = -A_struct  # a pattern matrix of -1's of action dim in each state row
            else:  # last horizon case
                A_trans_const[st_idx:st_idx+self.node_size,
                              ed_idx:ed_idx+self.node_size] = -torch.eye(self.node_size)
        # for last horizon the constraints are only in the past horizon

        A_dhsa = torch.cat([A_dpi_init, A_trans_const])
        b_dhsa = torch.zeros(A_dhsa.shape[0])
        b_dhsa[0:self.node_size] = dPi_init
        A_alpha = torch.matmul(A_dhsa, pathMatrix)
        # q_h_s is the optimization variable
        A_eq = torch.cat(
            [torch.ones(pathMatrix.shape[1]).reshape(1, -1), A_alpha]).numpy()
        b_eq = torch.cat(
            [torch.ones(1).reshape(-1), b_dhsa.reshape(-1)]).numpy()
        # A_eq = self.P.numpy()
        # b_eq = np.zeros(self.node_size)
        options_dict = {'presolve': True, 'dual_feasibility_tolerance': 1e-7,
                        'primal_feasibility_tolerance': 1e-6, 'disp': False}
        sol = opt.linprog(
            rew, bounds=[0, 1], A_eq=A_eq, b_eq=b_eq, options=options_dict, method='highs-ds')
        return sol

    def get_statewise_pathmatrix(self, pathMatrix):
        pathMatrix_s = torch.zeros(
            self.node_size*self.horizon, pathMatrix.shape[1])
        for h in range(self.horizon-1):
            for s_idx in range(self.node_size):
                pathMatrix_s[h*self.node_size + s_idx, :] = torch.sum(
                    pathMatrix[h*self.node_size*self.action_dim + s_idx*self.action_dim:h*self.node_size*self.action_dim + (s_idx+1)*self.action_dim, :], 0)
        pathMatrix_s[-self.node_size:, :] = pathMatrix[-self.node_size:, :]
        return 1.0*(pathMatrix_s > 0.1)

    def get_idx_action(self, h, s, path):
        if h < self.env_params["horizon"]-1:
            action = self.horizon_transition_graph.get_edge_data(
                path[h], path[h+1])['action']
            idx = h*self.node_size*self.action_dim + s*self.action_dim + action
        else:
            idx = h*self.node_size*self.action_dim + s
        # return self.horizon*s + h
        return idx

    def all_simple_paths(self):
        i = 0
        paths = dict()
        # This matrix dimention is (h,s) \times valid paths
        pathMatrix = torch.zeros((self.horizon*self.node_size, 1))
        paths_reward = []
        for h1 in range(self.horizon):
            for node1 in range(self.node_size):
                for h2 in range(self.horizon):
                    for node2 in range(self.node_size):
                        all_paths = nx.all_simple_paths(self.horizon_transition_graph, source=(
                            h1, node1), target=(h2, node2), cutoff=self.horizon-1)

                        for path in all_paths:
                            if len(path) == self.horizon:
                                paths[tuple(path)] = i
                                # print(i)
                                i += 1
                                paths_reward.append(
                                    self.weighted_submodular_return(path))
                                pathstates = torch.zeros(
                                    ((self.horizon-1)*self.node_size*self.action_dim + self.node_size, 1))
                                for hs in path:
                                    # record states visited by the path
                                    idx = self.get_idx_action(
                                        hs[0], hs[1], path)
                                    pathstates[idx] = 1
                                if (i == 1):
                                    print("in node")
                                    pathMatrix = pathstates
                                else:
                                    pathMatrix = torch.cat(
                                        [pathMatrix, pathstates], 1)
                            # print(pathMatrix.shape)
        return pathMatrix, paths_reward, paths

    def get_all_paths_act(self):
        i = 0
        paths = dict()
        # This matrix dimention is (h,s) \times valid paths
        pathMatrix = torch.zeros((self.horizon*self.node_size, 1))
        paths_reward = []
        for h in range(self.horizon):
            print(i)
            for node in range(self.node_size):
                paths_node = self.findPaths(
                    self.horizon_transition_graph, (h, node), self.horizon-1)
                # print(h, node, len(paths_node))
                for iter, path in enumerate(paths_node):
                    paths[tuple(path)] = i
                    # print(i)
                    i += 1
                    paths_reward.append(self.weighted_submodular_return(path))
                    pathstates = torch.zeros(
                        ((self.horizon-1)*self.node_size*self.action_dim + self.node_size, 1))
                    for hs in path:
                        # record states visited by the path
                        idx = self.get_idx_action(hs[0], hs[1], path)
                        pathstates[idx] = 1
                    if (node == 0 and h == 0 and iter == 0):
                        print("in node")
                        pathMatrix = pathstates
                    else:
                        pathMatrix = torch.cat([pathMatrix, pathstates], 1)
                    # print(pathMatrix.shape)
        return pathMatrix, paths_reward, paths

    def get_J_pi_A(self):
        i = 0
        paths = dict()
        # This matrix dimention is (h,s) \times valid paths
        pathMatrix = torch.zeros((self.horizon*self.node_size, 1))
        paths_reward = []
        for h in range(self.horizon):
            for node in range(self.node_size):
                paths_node = self.findPaths(
                    self.horizon_transition_graph, (h, node), self.horizon-1)
                # print(h, node, len(paths_node))
                for iter, path in enumerate(paths_node):
                    paths[tuple(path)] = i
                    # print(i)
                    i += 1
                    paths_reward.append(self.weighted_submodular_return(path))
                    pathstates = torch.zeros((self.horizon*self.node_size, 1))
                    for hs in path:
                        # record states visited by the path
                        idx = self.get_idx(hs[0], hs[1])
                        pathstates[idx] = 1
                    if (node == 0 and h == 0 and iter == 0):
                        # print("in node")
                        pathMatrix = pathstates
                    else:
                        pathMatrix = torch.cat([pathMatrix, pathstates], 1)
                    # print(pathMatrix.shape)
        return pathMatrix, paths_reward, paths

    def get_idx(self, h, s):
        # return self.horizon*s + h
        return h*self.node_size + s

    def findPaths2(self, G, u, n, excludeSet=None):
        if excludeSet == None:
            excludeSet = set([u])
        else:
            excludeSet.add(u)
        if n == 0:
            return [[u]]
        paths = [[u]+path for neighbor in G.neighbors(
            u) if neighbor not in excludeSet for path in self.findPaths2(G, neighbor, n-1, excludeSet)]
        excludeSet.remove(u)
        return paths

    def findPaths(self, G, u, n):
        '''https://stackoverflow.com/questions/28095646/finding-all-paths-walks-of-given-length-in-a-networkx-graph
        '''
        if n == 0:
            return [[u]]  # may be this is not require []
        paths = [[u]+path for neighbor in G.neighbors(
            u) for path in self.findPaths(G, neighbor, n-1) if u not in path]
        return paths

    def add_weights_to_node(self, val_matrix):
        for t in range(self.env_params["horizon"]):
            for i in range(self.env_size):
                self.horizon_transition_graph.nodes[(
                    t, i)]['weight'] = val_matrix[t*self.env_size + i]

    def get_traj(self, init_node, in_actions):
        curr_node = (0, init_node)
        traj = []
        traj.append(curr_node)
        for i in range(1, self.env_params['horizon']):
            for s, sd, action in self.transition_graph.edges(traj[-1][1], data=True):
                if action['action'] == in_actions[s]:
                    traj.append((i, sd))
        return traj

    def get_traj_policy(self, init_node, pi_s_a_h):
        curr_node = (0, init_node)
        traj = []
        traj.append(curr_node)
        for h in range(1, self.env_params['horizon']):
            for s, sd, action in self.transition_graph.edges(traj[-1][1], data=True):
                if action['action'] == torch.argmax(pi_s_a_h[s, :, h-1]).item():
                    traj.append((h, sd))
        return traj

    def find_best_traj(self, init_node):
        curr_node = (0, init_node)
        traj = []
        traj.append(curr_node)
        for i in range(1, self.env_params['horizon']):
            neighbors = list(
                self.horizon_transition_graph.neighbors(curr_node))
            weights = []
            for node in neighbors:
                weights.append(
                    self.horizon_transition_graph.nodes[node]['weight'])
            curr_node = neighbors[np.argmax(weights)]
            traj.append(curr_node)
        return traj

    def gen_random_traj(self, init_node):
        curr_node = init_node
        traj = []
        traj.append((0, curr_node))
        for i in range(1, self.env_params['horizon']):
            # neighbors = list(
            #     self.transition_graph.neighbors(curr_node))  # [0]  #
            neighbors = [0]
            curr_node = random.choice(neighbors)
            traj.append((i, curr_node))
        return traj

    def reset(self):
        self.state = torch.zeros((2, 1))
        self.reward = np.zeros((2, 1))
        self.done = False
        info = {}
        return self.state, self.reward[0], self.reward[1], self.done, info
    
    def __true_density_sampling(self):
        # torch.Tensor([0]).reshape(-1, 1)
        self.Fx_X = (torch.rand(2) * 10).reshape(-1, 2)
        self.Fx_Y = torch.zeros(self.Fx_X.shape[0], 1)
        self.Fx_model = SingleTaskGP(
            self.Fx_X, self.Fx_Y, covar_module=self.Fx_covar_module
        )
        self.Fx_model.covar_module.base_kernel.lengthscale = self.Fx_lengthscale
        self.Fx_model.likelihood.noise = self.Fx_noise
        density = self.Fx_model.posterior(self.gridV).sample().reshape(-1)
        if density.min() > -3:
            return density + 3
        else:
            return density + density.min()

    def true_density_sampling(self):
        # self.Fx_X = (torch.rand(2)*10).reshape(-1, self.env_dim)
        # self.Fx_Y = torch.zeros(self.Fx_X.shape[0], 1)
        # self.Fx_model = SingleTaskGP(
        #     self.Fx_X, self.Fx_Y, covar_module=self.Fx_covar_module)
        # self.Fx_model.covar_module.base_kernel.lengthscale = self.Fx_lengthscale
        # self.Fx_model.likelihood.noise = self.Fx_noise
        # density = self.Fx_model.posterior(
        #     self.VisuGrid).sample().reshape(-1, 1)
        # if density.min() > -3:
        #     meas_density = density + 3
        # else:
        #     meas_density = density + density.min()
        # self.Fx_model_cont = SingleTaskGP(
        #     self.VisuGrid, meas_density, covar_module=self.Fx_covar_module)

        return torch.zeros(self.gridV.shape[0]).reshape(-1, 1)

    def get_multilinear_gradient(self, dPi):
        grad = torch.zeros_like(dPi)
        for node in self.rew_graph.nodes:
            edges = self.rew_graph.edges(node)
            connected_nodes = [node]+[v for u, v in edges]
            # grad[node] = 1
            for u_node in connected_nodes:
                edges = self.coverage_graph.edges(u_node)
                Pu = [u_node] + [v for u, v in edges]
                factor = 1
                for x in Pu:
                    if x is not u_node:
                        factor *= (1-dPi[x])
                grad[node] += factor
        return grad

    def coverage_function(self, X):
        F = 0
        for u_node in self.transition_graph.nodes:  # [s_0,s_1,s_8,s_9]
            edges = self.coverage_graph.edges(u_node)
            # how can we reach this u_node from s,a is the actual question
            Pu = [u_node] + [v for u, v in edges]
            # which action from s lends us to pu
            factor = 1
            for s in Pu:
                # The covering of s can be at any horizon
                for t in range(self.env_params["horizon"]):
                    factor *= (1-X[s, t])
            F += (1-factor)*self.weight[u_node]  # (1/(u_node+1))
        return F

    def multi_linear_func_sym(self, dPi):
        subs_dict = dict(zip(self.X.ravel(), dPi.numpy().ravel()))
        return self.F.subs(subs_dict)

    def multi_linear_grad_sym(self, dPi):
        subs_dict = dict(zip(self.X.ravel(), dPi.ravel()))
        return self.DF.subs(subs_dict)

    def powerset(self, iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


if __name__ == '__main__':
    workspace = "subrl"
    with open(workspace + "/params/subrl.yaml") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    if params["env"]["generate"]:
        for i in range(0, 10):
            # save_path = workspace + "/experiments/" + datetime.today().strftime('%d-%m-%y') + \
            #     datetime.today().strftime(
            #         '-%A')[0:4] + "/environments/env_" + str(i) + "/"
            save_path = workspace + \
                "/environments/" + params["env"]["node_weight"] + "/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            env = GridWorld(
                env_params=params["env"], common_params=params["common"], visu_params=params["visu"], env_file_path=save_path + "env_" + str(i) + ".pkl")
    else:
        exp_name = params["experiment"]["name"]
        env_load_path = workspace + \
            "/experiments/20-08-22-Sat/environments/env_" + str(0) + "/"
        save_path = env_load_path + "/"

    # env = GridWorld(env_params=params["env"], common_params=params["common"],
    #                 visu_params=params["visu"], env_dir=save_path)
    # state, _, _, _, _ = env.reset()

    # for i in range(10):
    #     action1 = np.array(1).reshape(1)
    #     action2 = np.array(1).reshape(1)

    #     state, reward1, reward2, done, _ = env.step(action1, action2)
    #     print(state, reward1, reward2, done)
