import os
import glob
import numpy as np
from tensorpack import RNGDataFlow
from record_breakout import Recorder
import glob
from scipy import misc
import gym
from cv2 import resize

FRAME_HISTORY = 4
GAMMA = 0.99
TRAIN_TEST_SPLIT = 0.8

# Timon will pass me the key
GAME_NAMES = {
    'MontezumaRevenge-v0': 'revenge',
    'MsPacman-v0': 'mspacman',
    'SpaceInvaders-v0': 'spaceinvaders'
}

class Kurin_Reader():
    def __init__(self, record_folder, gym_game_name, data_frac=1.0):
        self.record_folder = record_folder
        self.gym_game_name = gym_game_name
        self.kurin_to_gym = self.get_kurin_to_gym_action_map()
        self.data_frac = data_frac

    def read_eps(self):
        eps_numbers = glob.glob(os.path.join(self.record_folder, 'screens', GAME_NAMES[self.gym_game_name], '*'))
        eps_numbers = [x.split('/')[-1] for x in eps_numbers]
        eps_numbers = eps_numbers[:int(self.data_frac*len(eps_numbers))]
        for eps_num in eps_numbers:
            full_eps_dict = {} # needs to have 'obs', 'act', 'rew'
            full_eps_dict['obs'] = self.read_obs(eps_num)
            full_eps_dict['act'], full_eps_dict['rew'] = self.read_act_reward(eps_num)
            #print np.prod(full_eps_dict['obs'].shape)
            #print full_eps_dict['obs'].dtype
            yield full_eps_dict

    def read_obs(self, eps_num): # [?, 84, 84, 3]
        obs = None
        num_screens = len(glob.glob(os.path.join(self.record_folder, 'screens', GAME_NAMES[self.gym_game_name], str(eps_num), '*png')))
        screens = [] # list of screens
        for i in range(1, num_screens+1): # not 0
            image = misc.imread(os.path.join(self.record_folder, 'screens', GAME_NAMES[self.gym_game_name], str(eps_num), str(i)+'.png'))
            image = resize(image, dsize = (84, 84))
            screens.append(np.expand_dims(image, axis=0)) 
        return np.concatenate(screens, axis=0)

    def read_act_reward(self, eps_num): # [[?, actions], [?, rewards]]
        acts, rewards = [[], []]
        with open(os.path.join(self.record_folder, 'trajectories', GAME_NAMES[self.gym_game_name], str(eps_num)+'.txt'), 'r') as f:
            f.readline() # ignoring headers
            f.readline() # ignoring headers
            f.readline() # ignoring headers
            for line in f:
                line = line.strip().split(',') # [frame,reward,score,terminal, action]
                line = [x.strip() for x in line]
                rewards.append(float(line[1]))
                acts.append(float(line[4])) 
                acts.append(self.kurin_to_gym[int(line[4])])
        return np.asarray(acts), np.asarray(rewards)
     

    def get_kurin_to_gym_action_map(self):
        kurin_to_gym = {} # keys are action numbers in Kurin. Values are action numbers in OpenAI Gym. This is Game Specific!
        # list of action meanings in Kurin
        kurin_action_meanings =  ['NOOP', 'FIRE','UP','RIGHT','LEFT','DOWN','UPRIGHT','UPLEFT',
                                  'DOWNRIGHT','DOWNLEFT','UPFIRE','RIGHTFIRE','LEFTFIRE','DOWNFIRE',
                                  'UPRIGHTFIRE','UPLEFTFIRE','DOWNRIGHTFIRE','DOWNLEFTFIRE']
        # list of action meanings for given game in Gym
        env = gym.make(self.gym_game_name)
        gym_action_meanings = env.unwrapped.get_action_meanings()
        for i in range(len(kurin_action_meanings)):
            try:
                ind = gym_action_meanings.index(kurin_action_meanings[i])
                kurin_to_gym[i] = ind
            except ValueError:
                kurin_to_gym[i] = gym_action_meanings.index('NOOP') # NOOP in gym
        return kurin_to_gym 


class KurinDataFlow(RNGDataFlow):
    """
    Produces [state, action, reward] of human demonstrations,
    state is 84x84x12 in the range [0,255], action is an int.
    """

    def __init__(self, mode, record_folder=None, gym_game_name=None, data_frac=1.0):
        """
        Args:
            train_or_test (str): either 'train' or 'test'
            shuffle (bool): shuffle the dataset
        """
        if record_folder is None:
            record_folder = '/Users/kalpit/Desktop/CS234/cs234_proj/spaceinvaders'
        if gym_game_name is None:
            gym_game_name = 'spaceinvaders'
        assert mode in ['train', 'test', 'all']
        self.mode = mode
        self.shuffle = mode in ['train', 'all']

        states = []
        actions = []
        rewards = []
        scores = []

        rec = Kurin_Reader(record_folder=record_folder, gym_game_name=gym_game_name, data_frac=data_frac)
        eps_counter = 0
        for eps in rec.read_eps():
            s = eps['obs']
            a = eps['act']
            r = eps['rew']

            # process states
            s = np.pad(s, ((FRAME_HISTORY-1,FRAME_HISTORY), (0,0), (0,0), (0,0)), 'constant')
            s = np.concatenate([s[i:-(FRAME_HISTORY-i)] for i in range(FRAME_HISTORY)], axis=-1)
            s = s[:-(FRAME_HISTORY-1)]
            states.append(s)
            
            # actions
            actions.append(a)

            # human score
            scores.append(np.sum(r))

            # process rewards just like in tensorpack
            R = 0
            r = np.sign(r)
            for idx in range(len(r)):
                R = r[idx] + GAMMA * R
                r[idx] = R
            rewards.append(r)

            eps_counter += 1
            print('eps_counter: %d' % eps_counter)

        self.avg_human_score = np.mean(scores)
        self.num_episodes = eps_counter
        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)

        num = self.size()
        if mode != 'all':
            idxs = list(range(self.size()))
            # shuffle the same way every time
            np.random.seed(1)
            np.random.shuffle(idxs)
            self.states = self.states[idxs]
            self.actions = self.actions[idxs]
            self.rewards = self.rewards[idxs]
            if mode == 'train':
                self.states = self.states[:int(TRAIN_TEST_SPLIT*num)]
                self.actions = self.actions[:int(TRAIN_TEST_SPLIT*num)]
                self.rewards = self.rewards[:int(TRAIN_TEST_SPLIT*num)]
            elif mode == 'test':
                self.states = self.states[int(TRAIN_TEST_SPLIT*num):]
                self.actions = self.actions[int(TRAIN_TEST_SPLIT*num):]
                self.rewards = self.rewards[int(TRAIN_TEST_SPLIT*num):]



    def size(self):
        return self.states.shape[0]

    def get_data(self):
        idxs = list(range(self.size()))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            state = self.states[k]
            action = self.actions[k]
            reward = self.rewards[k]
            yield [state, action, reward]
     


if __name__=='__main__':
    gym_game_name = 'SpaceInvaders-v0'
    data_frac = 1.0
    ##def KurinDataFlow(mode, record_folder=None, game_name=None, read_from_npy=True)
    #rdf = KurinDataFlow('train', 9, record_folder='/Users/kalpit/Desktop/CS234/cs234_proj/mspacman', game_name='mspacman')
    rdf = KurinDataFlow('train', gym_game_name=gym_game_name, data_frac=data_frac)
        

