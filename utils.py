# -*- coding: utf-8 -*-
"""
@author: Hanming Liu
"""


import torch


actions = ('NORTH', 'EAST', 'SOUTH', 'WEST')
indices = {'NORTH': 0, 'EAST': 1, 'SOUTH': 2, 'WEST': 3}

opposite_actions = {
    'NORTH': 'SOUTH', 
    'EAST' : 'WEST', 
    'SOUTH': 'NORTH', 
    'WEST' : 'EAST'
    }

neighboring_table = tuple((
     tile - 11 if (tile//11) > 0 else tile + 66, # NORTH
     tile + 1  if (tile%11) < 10 else tile - 10, # EAST
     tile + 11 if (tile//11) < 6 else tile - 66, # SOUTH
     tile - 1  if (tile%11)  > 0 else tile + 10  # WEST
     ) for tile in range(77))


def flatten(x: list):
    return [item for sublist in x for item in sublist]


def avg_of(x: list):
    if not x:
        return 0
    return sum(x) / len(x)


def to_tensor(x, gpu=True):
    return torch.tensor(x).cuda() if gpu else torch.tensor(x)


def action_of(index: int):
    return actions[index]


def index_of(action: str):
    return indices[action]


def opposite_of(action: str):
    return opposite_actions[action]


def neighbors_of(tile: int):
    return neighboring_table[tile]


def heads_of(geese):
    return [goose[0] for goose in geese if goose]


def bodies_of(geese):
    return flatten([goose[1:-1] for goose in geese if len(goose) > 2])


def tails_of(geese):
    return [goose[-1] for goose in geese if len(goose) > 1]


def split(index, geese):
    copy = geese[:]
    return copy.pop(index), copy


def get_action_mask(index, geese, prev_action, gpu=True):
    agent = geese[index]
    assert agent
    
    head = agent[0]
    walls = set(bodies_of(geese) + heads_of(geese))
    neighbors = neighbors_of(head)
    
    mask = torch.ones(4, dtype=torch.bool)
    for i in range(4):
        if neighbors[i] in walls:
            mask[i] = False
            
    if prev_action is not None:
        oppo_idx = index_of(opposite_of(prev_action))
        mask[oppo_idx] = False
        
    return mask.cuda() if gpu else mask


def is_eating(agent, move, food):
    assert agent
    
    head = agent[0]
    return neighbors_of(head)[move] in food
    
                
            
