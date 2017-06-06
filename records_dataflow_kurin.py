import os
import glob
import numpy as np
from tensorpack import RNGDataFlow
from record_breakout import Recorder
import glob
from scipy import misc
#from cv2 import resize

FRAME_HISTORY = 4
GAMMA = 0.99
TRAIN_TEST_SPLIT = 0.8

class Kurin_Reader():
    def __init__(self, record_folder, game_name):
        self.record_folder = record_folder
        self.game_name = game_name

    def read_eps(self):
        eps_numbers = glob.glob(os.path.join(self.record_folder, 'screens', self.game_name, '*'))
        eps_numbers = [x.split('/')[-1] for x in eps_numbers]
        for eps_num in eps_numbers:
            full_eps_dict = {} # needs to have 'obs', 'act', 'reward'
            full_eps_dict['obs'] = self.read_obs(eps_num)
            full_eps_dict['act'], full_eps_dict['reward'] = self.read_act_reward(eps_num)
            yield full_eps_dict

    def read_obs(self, eps_num): # [?, 84, 84, 3]
        obs = None
        num_screens = len(glob.glob(os.path.join(self.record_folder, 'screens', self.game_name, str(eps_num), '*png')))
        screens = [] # list of screens
        for i in range(1, num_screens+1): # not 0
            image = misc.imread(os.path.join(self.record_folder, 'screens', self.game_name, str(eps_num), str(i)+'.png'))
            #image = resize(image, dsize = (84, 84))
            screens.append(image)   
        return np.concatenate(image)

class RecordsDataFlow(RNGDataFlow):
    """
    Produces [state, action, reward] of human demonstrations,
    state is 84x84x12 in the range [0,255], action is an int.
    """

    def __init__(self, mode, num_actions, record_folder=None, game_name=None):
        """
        Args:
            train_or_test (str): either 'train' or 'test'
            shuffle (bool): shuffle the dataset
        """
        if record_folder is None:
            record_folder = '/Users/kalpit/Desktop/CS234/cs234_proj/spaceinvaders'
        if game_name is None:
            game_name = 'spaceinvaders'
        assert mode in ['train', 'test', 'all']
        self.mode = mode
        self.shuffle = mode in ['train', 'all']
        self.num_actions = num_actions

        states = []
        actions = []
        rewards = []
        scores = []

        rec = Kurin_Reader(record_folder=record_folder, game_name=game_name)
        eps_counter = 0
        for eps in rec.read_eps():
            s = eps['obs']
            a = eps['act']
            r = eps['rew']

            # check for right action space
            if (a>=self.num_actions).any():
                print('drop episode {}'.format(eps_counter))
                continue

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
    rdf = RecordsDataFlow('train', 4)
        
