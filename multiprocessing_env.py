import numpy as np
from multiprocessing import Process, Pipe

def worker(remote, parent_remote, env_fn_wrapper):
	parent_remote.close()
	env = env_fn_wrapper.x()
	while True:
		cmd, data = remote.recv()
		if cmd == "step":
			ob, reward, done, info = env.step(data)
			if done:
				ob = env.reset()
			remote.send((ob, reward, done, info))
		elif cmd == "reset":
			ob = env.reset()
			remote.send(ob)
		elif cmd == "reset_task":
			ob = env.reset_task()
			remote.send(ob)
		elif cmd == "close":
			remote.close()
			break
		elif cmd == "get_spaces":
			remote.send((env.observation_space, env.action_space))
		else:
			raise NotImplementedError

class VecEnv(object):

	def __init__(self, num_envs, observation_space, action_space):

		self.num_envs = num_envs
		self.observation_space = observation_space
		self.action_space = action_space

	def reset(self):
		pass

	def step_async(self, actions):
		pass

	def step_wait(self):
		pass

	def close(self):
		pass

	def step(self, actions):
		self.step_async(actions)
		return self.step_wait()

class CloudpickleWrapper(object):

	def __init__(self, x):
		self.x = x

	def __getstate__(self):
		import cloudpickle
		return cloudpickle.dumps(self.x)

	def __setstate__(self, ob):
		import pickle
		self.x = pickle.loads(ob)

class SubprocVecEnv(VecEnv):

	def __init__(self, env_fns, spaces = None):

		self.waiting = False
		self.closed = False
		nenvs = len(env_fns)
		self.envs = nenvs
		self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
		self.ps = [Process(target=worker, args=(work_remotes, remote, CloudpickleWrapper(env))) for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
		for p in self.ps:
			p.daemon = True
			p.start()

		for remote in self.work_remotes:
			remote.close()

		self.remotes[0].send(("get_spaces", None))
		observation_space, action_space = self.remotes[0].recv()
		VecEnv.__init__(self, len(env_fns), observation_space, action_space)


	def step_async(self, actions):
		for remote, action in zip(self.remotes, actions):
			remote.send(("step", action))
		self.waiting = True

	def step_wait(self):
		results = [remote.recv() for remote in self.remotes]
		self.waiting = False
		obs, rews, dones, infos = zip(*results)
		return np.stack(obs), np.stack(rews), np.stack(dones), infos

	def reset(self):
		for remote in self.remotes:
			remote.send(("reset", None))

		return np.stack([remote.recv() for remote in self.remotes])

	def reset_task(self):
		for remote in self.remotes:
			remote.send(("reset_task", None))
		return np.stack([remote.recv() for remote in self.remotes])

	def close(self):
		if self.closed:
			return
		if self.waiting:
			for remote in self.remotes:
				remote.recv()

			for remote in self.remotes:
				remote.send(("close", None))

			for p in self.ps:
				p.join()
				self.closed = True

	def __len__(self):
		return self.nenvs