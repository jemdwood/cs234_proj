import gym
from gym.wrappers import SkipWrapper
import cPickle as pick
# from gym.utils.replay_buffer import ReplayBuffer #IN CASE WE NEED IT LATER!!!
from gym.utils.json_utils import json_encode_np
from PIL import Image

import gym
import pygame
import sys
import time
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from collections import deque
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, VIDEORESIZE
from threading import Thread

try:
	matplotlib.use('GTK3Agg')
except Exception:
	pass


SKIP  = 0
RECORD_FILE = './records'
SCORE_THRESHOLD = 1000

def pickle(obj, f):
	# spreewalds galore
	try:
		if(type(obj) == type(list) or type(obj) == type(set)):
			obj = map(json_encode_np, obj)
		else:
			obj = json_encode_np(obj)
	except Exception:
		pass
	pick.dump(obj, f)

def unpickle(f):
	return pick.load(f)


class Recorder():
	def __init__(self, record_file = RECORD_FILE, immediate_flush = False, score_threshold = SCORE_THRESHOLD):
		self.record_file = record_file
		self.record_buffer = []	
		self.imm_flush = immediate_flush
		if not immediate_flush:
			self.current_buffer_score = 0
			self.sc_thresh = score_threshold

	def compress_obs():
		# http://stackoverflow.com/questions/10607468/how-to-reduce-the-image-file-size-using-pil
		pass


	def buffer_SARSD(self, prev_obs, obs, action, rew, env_done, info):
		SARSD = (prev_obs, action, rew, obs, env_done)
		if(self.imm_flush):
			with open(self.record_file, 'a') as f:
				pickle(SARSD, f)
		else:
			self.current_buffer_score += rew
			self.record_buffer.append(SARSD)

	def record(self):
		print('recording from buffer...')
		if not self.imm_flush:
			if len(self.record_buffer) > 0:
				if self.current_buffer_score >= self.sc_thresh:
					with open(self.record_file, 'a') as f:
						pickle(self.record_buffer, f)
					# or should this be the below? Do we want to reconstitute the buffer or
					# map(lambda x: pickle(x, self.record_file), self.record_buffer)
				else:
					print("score too low to bother recording")

		else:
			print("NOTE: Using immediate buffer flushing, do not use record pls")
			return
		self.current_buffer_score = 0







#
# Not ours, and not 100% sure what it's doing
def display_arr(screen, arr, video_size, transpose):
	arr_min, arr_max = arr.min(), arr.max()
	arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
	pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
	pyg_img = pygame.transform.scale(pyg_img, video_size)
	screen.blit(pyg_img, (0,0))

def record_game(env, record_file, frames_to_record = 1 , transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None):
	"""
	For our purposes, modify frames_to_record if you want to not record every single frame. The default value of 1 records every frame

	If you wish to plot real time statistics as you play, you can use
	PlayPlot. Here's a sample code for plotting the reward
	for last 5 second of gameplay.

		def callback(obs_t, obs_tp1, rew, done, info):
			return [rew,]
		env_plotter = EnvPlotter(callback, 30 * 5, ["reward"])

		env = gym.make("Pong-v3")
		play_game(env, callback=env_plotter.callback)


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
	recorder = Recorder(record_file, immediate_flush = False)
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
			recorder.record() #Records it all at the end of the montezuma episode
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
				print('Don\'t push too many keys you dick')
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



env = gym.make('MontezumaRevenge-v0')
wrapper = SkipWrapper(SKIP) # 0 = don't skip
env = wrapper(env)


record_game(env, RECORD_FILE, zoom=2)















