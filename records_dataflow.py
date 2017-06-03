import numpy as np
from tensorpack import RNGDataFlow
from record_breakout import Recorder

FRAME_HISTORY = 4
GAMMA = 0.99
TRAIN_TEST_SPLIT = 0.8

class RecordsDataFlow(RNGDataFlow):
    """
    Produces [state, action, reward] of human demonstrations,
    state is 84x84x12 in the range [0,255], action is an int.
    """

    def __init__(self, mode, record_folder=None):
        """
        Args:
            train_or_test (str): either 'train' or 'test'
            shuffle (bool): shuffle the dataset
        """
        if record_folder is None:
            record_folder = '/data_4/rl/breakout_records/'
        assert mode in ['train', 'test', 'all']
        self.mode = mode
        self.shuffle = mode in ['train', 'all']

        states = []
        actions = []
        rewards = []
        scores = []

        rec = Recorder(record_folder=record_folder)
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

        self.avg_human_score = np.mean(scores)
        self.num_episodes = eps_counter
        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)

        num = self.size()
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
