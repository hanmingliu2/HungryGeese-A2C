# -*- coding: utf-8 -*-
"""
@author: Hanming Liu
"""


import torch
from network import create_network
from utils import *


def create_state(observations, gpu=True):
    obs = observations[-1]
    index = obs['index']
    geese = obs['geese']
    food  = obs['food']
    
    agent, others = split(index, geese)
    assert agent

    state = torch.zeros(9, 77, dtype=torch.uint8).detach()
    
    head = agent[0]
    body = agent[1:-1]
    tail = agent[-1]
    
    state[0, food] = 1
    state[1, head] = 1
    state[2, body] = 1
    state[3, tail] = 1
    
    state[4, heads_of(others)]  = 1
    state[5, bodies_of(others)] = 1
    state[6, tails_of(others)]  = 1
    
    # Previous head position.
    # Helps the network recognizes the opposite action is invalid.
    if len(observations) > 1:
        geese = observations[-2]['geese']
        agent = geese[index]
        
        prev_head = agent[0]
        state[7, prev_head] = 1
    
    # May help the network recognize where the boundaries are.
    # https://ai.stackexchange.com/questions/11014/why-is-a-constant-plane-of-ones-added-into-the-input-features-of-alphago
    state[8].fill_(1)
    
    state = state.view(1, 9, 7, 11)
    return state.float().cuda() if gpu else state
        

net = create_network(training=False)
observations = []
def agent(obs, _):
    observations.append(obs)
    with torch.no_grad():
        state = create_state(observations)
        logits, _ = net(state)
        move = logits.argmax().item()
    return action_of(move)
