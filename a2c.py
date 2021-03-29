# -*- coding: utf-8 -*-
"""
@author: Hanming Liu
"""


from network import *
from utils import *
from agent import create_state

import torch
from torch.distributions.categorical import Categorical
from kaggle_environments import make

# Helps debugging back-propagation.
torch.autograd.set_detect_anomaly(True)

class Memory:
    def __init__(self):
        self.Q = []    # Actual returns.
        self.V = []    # Predicted returns.
        self.E = []    # Entropy terms.
        self.log_probs = []
        
        
    def add(self, Qs, Vs, Es, log_probs):
        self.Q.append(Qs)
        self.V.append(Vs)
        self.E.append(Vs)
        self.log_probs.append(log_probs)
        
        
    def clear(self):
        self.Q.clear()
        self.V.clear()
        self.E.clear()
        self.log_probs.clear()
        

class A2C:
    def __init__(self):
        self.network = create_network()
        self.memory = Memory()
        
        # Hyperparameters
        self.GAMMA  = 0.99     # Discount factor.
        self.ALPHA  = 0.001    # Learning rate.
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), 
                                          lr=self.ALPHA)
        
    def predict(self, obs):
        '''
        Output policy and value prediction given an input state.
        '''
        
        state = create_state(obs)
        logits, value = self.network(state)
        logits.squeeze_()
        value.squeeze_()
        return logits, value
    
    
    def run_episode(self):
        '''
        Collect experiences from an episode of self-plays. 
        '''
        
        observations = [[] for i in range(4)]
        actions = [[None] for i in range(4)]
        rewards = [[] for i in range(4)]
        
        entropy = [[] for i in range(4)]
        log_pbs = [[] for i in range(4)]
        values  = [[] for i in range(4)]
        
        env = make('hungry_geese', debug=False)
        frame = env.reset(num_agents=4)
        
        while any(entry['status'] == 'ACTIVE' for entry in frame):
            step  = frame[0]['observation']['step']
            food  = frame[0]['observation']['food']
            geese = frame[0]['observation']['geese']
            
            for i in range(4):
                agent = geese[i]
                if not agent:
                    continue
                
                obs = {'index': i, 'geese': geese[:], 'food' : food[:]}
                observations[i].append(obs)
                
                logits, value = self.predict(observations[i])
                
                # Mask invalid actions to boost learning speed.
                action_mask = get_action_mask(i, geese, actions[i][-1])
                logits = torch.where(action_mask, 
                                     logits, 
                                     to_tensor(-1e5))
                
                policy = Categorical(logits=logits)
                move = policy.sample()
                
                # Naive reward mechanism to encourage eating.
                # Feel free to change it to improve model performance.
                if is_eating(agent, move, food):
                    rewards[i].append(1)
                else:
                    rewards[i].append(0)
                    
                actions[i].append(action_of(move))
                entropy[i].append(policy.entropy())
                log_pbs[i].append(policy.log_prob(move))
                values[i].append(value)

            # Next frame
            frame = env.step(tails_of(actions))
        
        # Episode is over.
        # Assign final reward for each agent.
        for i in range(4):
            score = frame[i]['reward']
            turns, length = divmod(score, 100)
            
            # Encourage surviving.
            rewards[i][-1] += turns/200
            
            # Ensure shapes are consistent.
            assert len(rewards[i]) == len(log_pbs[i]) == len(values[i])
    
        # Calculate actual returns.
        Q = []
        for i in range(4):
            length = len(rewards[i])
            returns = torch.zeros(length).detach()
            val = 0
            for t in reversed(range(length)):
                val = rewards[i][t] + self.GAMMA * val
                returns[t] = val
                
            Q.append(returns)
        
        # Flatten training data.
        Q = torch.hstack(Q).detach()
        V = torch.hstack(flatten(values))
        E = torch.hstack(flatten(entropy))
        log_probs = torch.hstack(flatten(log_pbs))
        
        # Again ensure shapes of training data are consistent.
        assert Q.shape == V.shape == E.shape == log_probs.shape
        
        self.memory.add(Q, V, E, log_probs)


    def learn(self):
        Q = torch.hstack(self.memory.Q).cuda()
        V = torch.hstack(self.memory.V)
        E = torch.hstack(self.memory.E)
        log_probs = torch.hstack(self.memory.log_probs)
        
        A = Q - V
        actor_loss = (-log_probs * A).mean()
        critic_loss = 0.5 * A.square().mean()
        entropy_loss = 0.001 * E.sum()
        loss = actor_loss + critic_loss + entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        torch.save(self.network.state_dict(), 'ZZGooseNet.pt')
        self.memory.clear()
        
        return loss.item()
        
            
if __name__ == '__main__':
    agent = A2C()
    for i in range(4000):
        agent.run_episode()
        if (i+1) % 4 == 0:
            loss = agent.learn()
            print('Training loss for episode %d to %d is: %.5f' % (i-3, i+1, loss))