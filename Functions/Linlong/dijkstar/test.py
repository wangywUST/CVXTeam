# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:15:39 2018

@author: lwuag
"""
import sys
sys.path.append("Functions/Linlong/dijkstar")
from graph import *
from algorithm import *

graph = Graph()
graph.add_edge(1, 2, {'cost': 1})
graph.add_edge(2, 3, {'cost': 2})
graph.add_node(4)
cost_func = lambda u, v, e, prev_e: e['cost']
path = find_path(graph, 1, 3, cost_func=cost_func)
#graph.dump(path)