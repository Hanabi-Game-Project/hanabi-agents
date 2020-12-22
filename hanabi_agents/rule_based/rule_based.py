from .ruleset import Ruleset
from hanabi_learning_environment import pyhanabi_pybind as pyhanabi
import timeit
import numpy as np


class RulebasedAgent():

    def __init__(self, rules, reward_shaper):
        self.rules = rules
        self.totalCalls = 0
        self.histogram = [0 for i in range(len(rules)+1)]
        self.reward_shaper = reward_shaper


    def get_move(self, observation):
        if observation.current_player_offset == 0:
            for index, rule in enumerate(self.rules):
                action = rule(observation)
                if action is not None:
                    self.histogram[index] += 1
                    self.totalCalls += 1
                    return action
            self.histogram[-1] += 1
            self.totalCalls += 1
            return Ruleset.legal_random(observation)
        return None

    def explore(self, observations):
        actions = pyhanabi.HanabiMoveVector()
        for observation in observations:
            actions.append(self.get_move(observation))
        return actions

    def exploit(self, observations):
        return self.explore(observations)

    def requires_vectorized_observation(self):
        return False

    def add_experience(self, otm1, atm1, rt, ot, tt):
        pass

    def update(self):
        pass
    
    def shape_rewards(self, observations, moves):
        if self.reward_shaper is not None:
            shaped_rewards, shape_type = self.reward_shaper.shape(observations, 
                                                                  moves)
            return np.array(shaped_rewards), np.array(shape_type)

        return (np.zeros((len(observations), )), np.zeros((len(observations),)))
    
    def create_stacker(self, obs_len, n_states):
        return None

    def save_weights(self, a, b):
        pass