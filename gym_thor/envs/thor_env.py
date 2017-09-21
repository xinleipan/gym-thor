import gym
import copy
import sys
import os
import platform
import robosims
import json
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from PIL import Image as Image
from scipy import misc
import pickle as pkl
import networkx as nx
import pdb
import random
 
class ThorEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    num_env = 0

    def __init__(self):
        ''' action space definition '''
        self.actions = (0,1,2,3,4,5,6,7,8)
        self.inv_actions = (0,2,1,4,3,6,5,8,7)
        self.action_space = spaces.Discrete(9)
        self.actions_dict = {0: 'Stay', 1: 'MoveLeft', 2:'MoveRight', 3:'MoveAhead',
            4:'MoveBack', 5:'LookUp', 6:'LookDown', 7:'RotateRight', 8:'RotateLeft'}
        
        ''' state space definition '''
        self.obs_shape = [128, 128, 3]
        self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_shape)
        self.observation = np.random.randn(128, 128, 3) 
        self.agent_state = [0,0,0,0,0,0,0]
        self.agent_start_state = [-1.0850000381469727, 0.9799996018409729, -2.678999900817871, 0.0, 90.0, 0.0, 330.0]
        self.agent_target_state = [-2.3350000381469727, 0.9799996614456177, -2.428999900817871, 0.0, 270.0, 0.0, 330.0]
        file_path = os.path.dirname(os.path.realpath(__file__))
        sys.path.append(file_path)
        
    def start(self, env_path, visited_path=None, G_path=None):
        ''' initialize system state '''
        system_name = platform.system()
        if system_name.lower()[:5] == 'linux':
            self.envs = robosims.controller.ChallengeController(
                unity_path=env_path+'thor-201705011400-Linux64',
                x_display="0.0")
        else:
            self.envs = robosims.controller.ChallengeController(
                unity_path=env_path+'thor-201705011400-OSXIntel64.app/Contents/MacOS/thor-201705011400-OSXIntel64')
        self.target = None
        with open(os.path.join(env_path,'thor-challenge-targets/targets-train.json')) as f:
            t = json.loads(f.read())
            for target in t:
                if target['sceneName'] == 'FloorPlan19':
                    self.target = target
                    break
        if self.target is None:
            sys.exit('no target state!')
        if visited_path is not None:
            self.visited = pkl.load(open(visited_path, 'rb'))
        else:
            self.visited = None
        if G_path is not None:
            self.G = pkl.load(open(G_path, 'rb'))
        else:
            self.G = None

        ''' initialize observation '''
        self.envs.start()
        self.observation = self._reset()
 
        ''' agent state: start, target, current state '''
        self.agent_start_state = copy.deepcopy(self.agent_state)
        self.agent_target_state = list(self.get_target(self.target))

        ''' set other parameters '''
        self.restart_once_done = True
        self.verbose = False

    def randomize_env(self, seed):
        random.seed(seed)
        if self.visited is None or self.G is None:
            sys.exit('please set your visited and G file!')
        else:
            num_nodes = len(self.visited.nodes)
            random_number = random.randint(0, num_nodes-1)
            self._reset()
            new_start_node = list(self.visited.nodes[random_number].coord)
            success = self.change_start_state(new_start_node)
            return success

    def _step(self, act):
        if type(act) == type(1):
            event = self.envs.step(action=dict(action=self.actions_dict[act]))
        else:
            event = self.envs.step(action=dict(action=act))
        img = event.frame
        this_state = list(self.get_coord(event))
        if np.sum(np.abs(np.array(this_state) - np.array(self.agent_target_state))) <= 0.01:
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        if event.metadata['lastActionSuccess']:
            success = True
        else:
            success = False
        self.observation = self.wrap(img)
        self.agent_state = this_state
        return (self.observation, reward, done, success)
            
    def _reset(self):
        self.envs.initialize_target(self.target)
        for act in self.actions_dict.keys():
            if act > 0:
                event = self.envs.step(action=dict(action=self.actions_dict[act]))
                if event.metadata['lastActionSuccess']:
                    event = self.envs.step(action=dict(action=self.actions_dict[act+(act%2!=0)-(act%2==0)]))
                    break
        self.agent_state = list(self.get_coord(event))
        self.observation = self.wrap(event.frame)
        if self.same_state(self.agent_state, self.agent_start_state) == False:
            curr_nd = Node(self.agent_state)
            start_nd = Node(self.agent_start_state)
            curr_name = self.visited.find_node(curr_nd)
            start_name = self.visited.find_node(start_nd)
            (_, actions) = find_all_path(self.G, self.visited, curr_name, start_name)
            for act in actions[0]:
                self.observation,_,_,_ = self._step(act) 
            if self.same_state(self.agent_state, self.agent_start_state) == False:
                sys.exit('cannot reset!')
        return self.observation

    def _render(self, mode='human', close=False):
        pass

    def same_state(self, state1, state2):
        state1 = np.array(state1)
        state2 = np.array(state2)
        if np.array_equal(state1, state2):
            return True
        else:
            return False
    
    def change_start_state(self, sp):
        if self.visited is None or self.G is None:
            print('You cannot change start state!')
            return False
        else:
            sp_nd = Node(list(sp))
            sp_nd_name = self.visited.find_node(sp_nd)
            current_state = self.get_agent_state()
            current_nd = Node(list(current_state))
            current_name = self.visited.find_node(current_nd)
            if sp_nd_name == -1 or current_name == -1:
                sys.exit('cannot change start state! didnt find it')
            else:
                (_, actions) = find_all_path(self.G, self.visited, current_name, sp_nd_name)
                for act in actions[0]:
                    _ = self._step(act)
                if self.same_state(self.agent_state, sp):
                    self.agent_start_state = copy.deepcopy(self.agent_state)                 
                    return True
                else:
                    return False
                     
    def change_target_state(self, tg):
        if self.visited is None or self.G is None:
            print('You cannot change goal state')
            return False
        else:
            tg_nd = Node(list(tg))
            tg_name = self.visited.find_node(tg_nd)
            if tg_name == -1:
                return False
            else:
                self.agent_target_state = copy.deepcopy(tg)
                self._reset()
                return True
                
    def get_agent_state(self):
        return self.agent_state
    
    def get_start_state(self):
        return self.agent_start_state

    def get_target_state(self):
        return self.agent_target_state

    def _jump_to_state(self, to_state):
        if self.visited is None or self.G is None:
            print('You cannot change state!')
            return (self.observation, 0, False, False)
        else:
            current_st = self.get_agent_state()
            current_nd = Node(list(current_st))
            current_name = self.visited.find_node(current_nd)
            to_nd = Node(list(to_state))
            to_name = self.visited.find_node(to_nd)
            if to_name == -1:
                print('You cannot jump to an non-existant state!')
                return (self.observation, 0, False, False)
            else:
                (_, actions) = find_all_path(self.G, self.visited, current_name, to_name)
                for act in actions[0]:
                    self.observation, reward, done, success = self._step(act)
                if self.same_state(self.agent_state, to_state) == False:
                    sys.exit('jump state fail!')
                else:
                    return (self.observation, reward, done, success) 
                   
    def jump_to_state(self, to_state):
        (a,b,c,d) = self._jump_to_state(to_state)
        return (a,b,c,d)

    def get_coord(self, event):
        x,y,z = event.metadata['agent']['position']['x'], \
                event.metadata['agent']['position']['y'], \
                event.metadata['agent']['position']['z']
        rx, ry, rz = event.metadata['agent']['rotation']['x'], \
                     event.metadata['agent']['rotation']['y'], \
                     event.metadata['agent']['rotation']['z']
        h = event.metadata['agent']['cameraHorizon']
        return list((x,y,z,rx,ry,rz,h))

    def wrap(self, img):
        img = Image.fromarray(img)
        size = 128, 128
        img.thumbnail(size)
        res_img = np.array(img)
        return res_img

    def get_target(self, target):
        (t_x,t_y,t_z,t_rx,t_ry,t_rz,t_h)=(target['targetPosition']['x'], target['targetPosition']['y'], target['targetPosition']['z'], 0.0, target['targetAgentRotation'], 0.0,target['targetAgentHorizon']) 
        return list((t_x,t_y,t_z,t_rx,t_ry,t_rz,t_h))


def find_all_path(G, visited, start_point, end_point):
    paths = []
    actions = []
    l, first_path = nx.bidirectional_dijkstra(G, start_point, end_point)
    finished = False
    paths.append(first_path)
    actions_list = []
    G2 = G.copy()
    for p in range(len(first_path) - 1):
        this_act = visited.edges[first_path[p], first_path[p+1]][0]
        if len(this_act) > 1:
            sys.exit('wrong action!')
        actions_list.append(visited.edges[first_path[p], first_path[p+1]][0][0])
    actions.append(actions_list)
    path_idx = -1
    while (finished == False):
        path_idx += 1
        current_path = paths[path_idx]
        len_paths = len(paths)
        for k in range(len(current_path)-1):
            G.remove_edge(current_path[k], current_path[k+1])
            try:
                this_l, this_path = nx.bidirectional_dijkstra(G, start_point, end_point)
                if this_l == l:
                    if this_path not in paths:
                        paths.append(this_path)
                        actions_list = []
                        for p in range(len(this_path) - 1):
                            this_act = visited.edges[this_path[p], this_path[p+1]][0]
                            if len(this_act) > 1:
                                sys.exit('wrong action!')
                            actions_list.append(visited.edges[this_path[p], this_path[p+1]][0][0])
                        actions.append(actions_list)
            except:
                pass
            G.add_edge(current_path[k], current_path[k+1])
        if len(paths) <= len_paths and path_idx == len(paths) - 1:
            finished = True

    return paths, actions

class Node(object):
    count = 0
    def __init__(self, coord, name=None, thre=0.01):
        self.coord = np.array(coord)
        self.thre = thre
        if name == None:
            self.node_name = Node.count
            Node.count += 1
        else:
            self.node_name = name

    def compare(self, B):
        s = np.sum(np.abs(self.coord - B.coord))
        if s >= self.thre:
            return False
        else:
            return True

class network(object):
    def __init__(self):
        self.nodes = []
        self.edges = {}
    def add_node(self, node):
        self.nodes.append(node)
    def node_exist(self, node):
        for n in self.nodes:
            if n.compare(node):
                return True
        return False
    def find_node(self, node):
        for n in self.nodes:
            if n.compare(node):
                return n.node_name
        return -1
    def add_edge(self, node1, node2, action, extra=False):
        if extra == False:
            if (node1.node_name, node2.node_name) in self.edges:
                self.edges[(node1.node_name, node2.node_name)][0] = action
            else:
                self.edges[(node1.node_name, node2.node_name)] = {}
                self.edges[(node1.node_name, node2.node_name)][0] = action
        else:
            len_key = 0
            try:
                len_key = len(self.edges[(node1.node_name, node2.node_name)])
            except:
                pass
            if len_key == 0:
                self.edges[(node1.node_name, node2.node_name)] = {}
            self.edges[(node1.node_name, node2.node_name)][len_key] = action
            
def update_dict(dest, src):
    res = copy.deepcopy(dest)
    largest = -1
    for k in res.keys():
        if k > largest:
            largest = k
    for key in src.keys():
        largest += 1
        res[largest] = src[key]
    return res
