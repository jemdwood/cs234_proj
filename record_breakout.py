import gym
from gym.wrappers import SkipWrapper
import pickle as pick
# from gym.utils.replay_buffer import ReplayBuffer #IN CASE WE NEED IT LATER!!!
from gym.utils.json_utils import json_encode_np
from PIL import Image

import gym
import pygame
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import random
import io
import os

import numpy as np
from collections import deque
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, VIDEORESIZE
from threading import Thread
import cv2

from gym import spaces
from viewer import SimpleImageViewer
from collections import deque

# have python3 map behave like python2
import six
from itertools import starmap

def map(func, *iterables):
    zipped = six.moves.zip_longest(*iterables)
    if func is None:
        # No need for a NOOP lambda here
        return zipped
    return list(starmap(func, zipped))

RECORD_EVERY  = 1 #record every n frames (should be >= 1)
SCORE_THRESHOLD = 15 #reasonably hard to achieve score. However the score is honestly oddly set up
HORIZ_DOWNSAMPLE =  1 # Leave at 1. Other values and you can't see some thin, but critical, parts of the environment
VERT_DOWNSAMPLE = 1 #1 or 2. I find it harder to do the laser gates when set to 2, but should theoretically be possible
SPEED = 0 # 0 or 1 at most. I find 1 difficult
FPS = 10

RECORD_FILE = './records.txt'
RECORD_FOLDER = './breakout_records/'
# FILE_EPISODE_DIVIDER = None#'\n<end_eps>----<end_eps>\n'


def downsample(state):
    # state = state[:195]  # crop
    # state = state[::VERT_DOWNSAMPLE,::HORIZ_DOWNSAMPLE] # downsample by factor of 2
    state = cv2.resize(state, (84, 84))
    return state.astype(np.uint8)

class PreproWrapper(gym.Wrapper):

    def __init__(self, env, prepro, shape, high=255):
        """
        Args:
            env: (gym env)
            prepro: (function) to apply to a state for preprocessing
            shape: (list) shape of obs after prepro
            overwrite_render: (bool) if True, render is overwriten to vizualise effect of prepro
            grey_scale: (bool) if True, assume grey scale, else black and white
            high: (int) max value of state after prepro
        """
        super(PreproWrapper, self).__init__(env)
        self.viewer = None
        self.prepro = prepro
        self.observation_space = spaces.Box(low=0, high=high, shape=shape)
        self.high = high


    def _step(self, action):
        """
        Overwrites _step function from environment to apply preprocess
        """
        obs, reward, done, info = self.env.step(action)
        self.obs = self.prepro(obs)
        return self.obs, reward, done, info


    def _reset(self):
        self.obs = self.prepro(self.env.reset())
        return self.obs


    def _render(self, mode='human', close=False):
        """
        Overwrite _render function to vizualize preprocessing
        """
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self.obs
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)


'''
General class used to record and retrieve episodes in a numpy format. Unless immediate_flush is set to true,
the general usage of this class should follow:
for each episode:
	while episode is not over, for each SARSD tuple:
		Recorder.buffer_SARSD(...)
	Recorder.record_eps()
... do stuff
for episode in Recorder.read_episode():
	do stuff on episode
'''
class Recorder():
	def __init__(self, record_file = RECORD_FILE, immediate_flush = False, score_threshold = SCORE_THRESHOLD):
		self.record_file = record_file
		self.SARSD_keys = ['prev_obs', 'obs', 'act', 'rew', 'done']
		self.record_buffer = dict()
		for key in self.SARSD_keys:
			self.record_buffer[key] = []
		self.imm_flush = immediate_flush
		if not immediate_flush:
			self.current_buffer_score = 0
			self.sc_thresh = score_threshold

	'''
	Buffers a SARSD tuple but does NOT write to a file unless immediate flushing was set to true
	'''
	def buffer_SARSD(self, prev_obs, obs, action, rew, env_done, info):
		obs = obs.astype(np.int8)
		#print(obs.shape)
		#print(100928.0/sys.getsizeof(obs), 'x improved')
		#prev_obs = prev_obs.astype(np.float32) #float32 is faster on gpu, supposedly
		SARSD = (prev_obs, action, rew, obs, env_done)
		if(self.imm_flush):
			with open(self.record_file, 'a') as f:
				pickle(SARSD, f) #immediate flushing pickles objects, pls don't use
		else:
			self.current_buffer_score += rew
			#self.record_buffer.append(SARSD)
			self.record_buffer['prev_obs'].append(prev_obs)
			self.record_buffer['obs'].append(obs)
			self.record_buffer['act'].append(action)
			self.record_buffer['rew'].append(rew)
			self.record_buffer['done'].append(env_done)


	def rec_file_path(self, rec_key):
		if not os.path.exists(RECORD_FOLDER):
			os.mkdir(RECORD_FOLDER)
		return RECORD_FOLDER + rec_key + '_record.txt'

	def get_key_from_path(self, fp):
		file_name = fp.split('/')[-1]
		key = file_name.split('_record.txt')[0]
		return key

	'''
	Record Epsidode
	Call to actually store the buffered episode in the record file. This should be called
	at the end of every episode (unless the recorder is configured to immediately flush data).
	'''
	def record_eps(self):
		if not self.imm_flush:
			if len(self.record_buffer['rew']) > 0:
				print('recording from buffer...')
				if self.current_buffer_score >= self.sc_thresh:
					for key in self.SARSD_keys:
						with open(self.rec_file_path(key), 'a') as f:
							obj = np.array(self.record_buffer[key])
							np.save(f, obj)
							#f.write(FILE_EPISODE_DIVIDER) #TODO???
							print('%s recorded' %(key))
				else:
					print("score too low to bother recording -- score = %i" % (self.current_buffer_score))
				print('...emptying buffer')
				for key in self.SARSD_keys:
					del self.record_buffer[key][:]
		else:
			print("NOTE: Using immediate buffer flushing, do not use record pls")
			return
		self.current_buffer_score = 0

	'''
	Does not support immediate flushing. Immediate flushing should really just be used for debugging.
	Returns: a generator over dicts with self.SARSD_keys as the keys, each mapping to their respective data
	Usage:
	for episode in Recorder.read_eps():
		rewards = episode['rew']
		episode_score = sum(rewards)
		for t in range(len(rewards)):
			SARSD_t = map(lambda key: x[key][t], Recorder.SARSD_keys)
	'''
	def read_eps(self):
		file_names = map(self.rec_file_path, self.SARSD_keys)
		file_d = dict()
		map(file_d.update, map(lambda fn: {self.get_key_from_path(fn): io.open(fn, 'rb')} , file_names))

		while True:
			full_eps_dict = dict()
			for key in self.SARSD_keys:
				try:
					eps_data = np.load(file_d[key])
					full_eps_dict[key] = eps_data
				except IOError as e:
					map(lambda x: x.close(), file_d.values())
					return #read is finished
			yield full_eps_dict







#
# Not ours, and not 100% sure what it's doing. Copied from utils.play
def display_arr(screen, arr, video_size, transpose):
	arr_min, arr_max = arr.min(), arr.max()
	arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
	pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
	pyg_img = pygame.transform.scale(pyg_img, video_size)
	screen.blit(pyg_img, (0,0))

def record_game(env, record_file, frames_to_record = RECORD_EVERY , transpose=True, fps=FPS, zoom=None, callback=None, keys_to_action=None):
	"""
	For our purposes, modify frames_to_record if you want to not record every single frame. The default value of 1 records every frame.
	This method was largely copied from gym.utils.play however it has some modifications to record the data

	Arguments
	---------
	env: gym.Env
		Environment to use for playing.
	transpose: bool
		If True the output of observation is transposed.
		Defaults to true.
	fps: int
		Maximum number of steps of the environment to execute every second.
		Defaults to 30.
	zoom: float
		Make screen edge this many times bigger
	callback: lambda or None
		Callback if a callback is provided it will be executed after
		every step. It takes the following input:
			obs_t: observation before performing action
			obs_tp1: observation after performing action
			action: action that was executed
			rew: reward that was received
			done: whether the environemnt is done or not
			info: debug info
	keys_to_action: dict: tuple(int) -> int or None
		Mapping from keys pressed to action performed.
		For example if pressed 'w' and space at the same time is supposed
		to trigger action number 2 then key_to_action dict would look like this:

			{
				# ...
				sorted(ord('w'), ord(' ')) -> 2
				# ...
			}
		If None, default key_to_action mapping for that env is used, if provided.
	"""
	recorder = Recorder()
	obs_s = env.observation_space
	assert type(obs_s) == gym.spaces.box.Box
	assert len(obs_s.shape) == 2 or (len(obs_s.shape) == 3 and obs_s.shape[2] in [1,3])

	if keys_to_action is None:
		if hasattr(env, 'get_keys_to_action'):
			keys_to_action = env.get_keys_to_action()
		elif hasattr(env.unwrapped, 'get_keys_to_action'):
			keys_to_action = env.unwrapped.get_keys_to_action()
		else:
			assert False, env.spec.id + " does not have explicit key to action mapping, " + \
						  "please specify one manually"
	relevant_keys = set(sum(map(list, keys_to_action.keys()),[]))

	if transpose:
		video_size = env.observation_space.shape[1], env.observation_space.shape[0]
	else:
		video_size = env.observation_space.shape[0], env.observation_space.shape[1]

	if zoom is not None:
		video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

	video_size = (video_size[0], video_size[1])
	pressed_keys = []
	running = True
	env_done = True

	screen = pygame.display.set_mode(video_size)
	clock = pygame.time.Clock()

	while running:
		if env_done:
			env_done = False
			obs = env.reset()
			recorder.record_eps() #Records it all at the end of the montezuma episode
		else:
			try:
				action = keys_to_action[tuple(sorted(pressed_keys))]
				prev_obs = obs
				obs, rew, env_done, info = env.step(action)

				if callback is not None:
					callback(prev_obs, obs, action, rew, env_done, info)

				time = clock.get_rawtime()
				if(time % frames_to_record == 0):
					recorder.buffer_SARSD(prev_obs, obs, action, rew, env_done, info)
			except KeyError:
				print('Don\'t push too many keys guys')
		if obs is not None:
			if len(obs.shape) == 2:
				obs = obs[:, :, None]
			if obs.shape[2] == 1:
				obs = obs.repeat(3, axis=2)
			display_arr(screen, obs, transpose=transpose, video_size=video_size)

		# process pygame events
		for event in pygame.event.get():
			# test events, set key states
			if event.type == pygame.KEYDOWN:
				if event.key in relevant_keys:
					pressed_keys.append(event.key)
				elif event.key == 27:
					running = False
			elif event.type == pygame.KEYUP:
				if event.key in relevant_keys:
					pressed_keys.remove(event.key)
			elif event.type == pygame.QUIT:
				running = False
			elif event.type == VIDEORESIZE:
				video_size = event.size
				screen = pygame.display.set_mode(video_size)
				print(video_size)

		pygame.display.flip()
		clock.tick(fps)
	pygame.quit()


if __name__ == '__main__':
	env = gym.make('Breakout-v0')
	wrapper = SkipWrapper(SPEED) # 0 = don't skip
	env = wrapper(env)
	env = PreproWrapper(env, prepro=lambda x: downsample(x), shape=(105, 80, 3))

	record_game(env, RECORD_FILE, zoom=4)















