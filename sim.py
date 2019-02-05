import numpy as np
import pygame
import yaml
import copy
import time
import sys
from operator import itemgetter
import math
import gym
from gym.spaces import Discrete, Box
from gym.envs.registration import EnvSpec

'''
 The code inside this tag can be configured by changing the values. Some of the such
 variables can be seen in config files and some are not. This is done, to make sure
 default values are used. However, the code will work even if these values are changed
 to some non-default values.
 # ------ Configurable ------- #


'''
class sim(gym.Env):

	def __init__(self, config):

		self.render = config["render"]
		self.lanes = config["lanes"]
		self.fps = config["fps"]
		self.time_period = config["time-period"]
		self.max_velocity = config["max-velocity"]
		self.view_size = config["view-size"] # add condition for max and min view-size
		self.cell_size = config["cell-size"] # add contion for max and min cell-size
		self.config = config
		self.info_dict = {}
		self.info_dict["collide_count"] = 0

		if config["use-seed"]:
			np.random.seed(config["seed"])

		if (self.view_size % self.cell_size) != 0:
			print("View size should be divisible by cell size")
			sys.exit(-1) 
		
		self.num_cars = config["rows"] # add max and min check condition

		if self.num_cars < 5:
			print("At least 5 rows are required.")
			sys.exit(-2)

		if self.lanes < 2:
			print("At least two lanes are required.")
			sys.exit(-1)

		self.scale = 8

		cal_travel_pixels = self.max_velocity * self.time_period * self.scale
		
		if cal_travel_pixels > self.scale * self.cell_size:
			print("Invalid max speed, Decrease it to make it compitable with current scale")
			sys.exit()
		
		#self.scale = 8 
		self.init_vehicle_information()
		self.init_locs()
		self.action_map = {0: "acc", 1: "dec", 2: "nothing", 3: "lane change", None: "None"}

		self.densities = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
		self.screen_height = self.car_length_with_min_distance_pixels * self.num_cars

		
		self.init_occupancy()
		self.init_gym()

		if self.render:
			self.init_render()

	def init_gym(self):

		self.observation_space = Box(-float("inf"), float("inf"), shape=(self.lanes * self.view_size + 2, ), dtype=np.float) # +2 for agent speed and agent lane
		self.action_space = Discrete(3)

	def init_vehicle_information(self):

		# ------ Configurable ------- #
		self.car_length = 4
		self.car_length_pixel = self.scale * self.car_length
		#print("Car length set to : %.2f px"%(self.car_length_pixel))
		self.car_length_with_min_distance = 6 
		self.car_length_with_min_distance_pixels = self.scale * self.car_length_with_min_distance
		self.car_width = 24
		#print("Car width set to : %.2f px"%(self.car_width))	
		self.a = self.config["car-acc"]
		self.b = self.config["car-dec"]
		#print("Car acc : ", self.a)
		#print("Car dec : ", self.b)
		# ------ Configurable ------- #

	def init_render(self):
		self.screen_width = 350
		self.lane_width = 50

		self.color_green = (107,142,35)
		self.road_color = (97, 106, 107)
		self.color_black = (0, 0, 0)
		self.color_white = (255, 255, 255)
		self.color_red = (128,0,0)
		self.color_dark_green = (0,100,0)


		pygame.init()
		pygame.font.init()


		pygame.display.set_caption('Traffic Simulator')
		self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
		self.clock = pygame.time.Clock()
		self.font = pygame.font.SysFont("Comic Sans MS", 20)

		#self.car_image = pygame.image.load('Images/agent_car.png').convert_alpha()
		#self.car_image = pygame.transform.scale(self.car_image,(self.car_width, self.car_length_pixel))

		#self.other_car_image = pygame.image.load('Images/other_cars.png').convert_alpha()
		#self.other_car_image = pygame.transform.scale(self.other_car_image,(self.car_width, self.car_length_pixel))
	
	def init_locs(self):

		self.init_pos_list = []
		for i in range(self.num_cars):
			pixel_val = (i * self.car_length_with_min_distance_pixels) + self.car_length_with_min_distance_pixels
			pixel_val -= self.car_length_with_min_distance_pixels / 2
			pixel_val -= self.car_length_pixel / 2
			self.init_pos_list.append(pixel_val)

		self.agent_loc_pixels = self.init_pos_list[-2]

	def get_loc(self, loc):

		return (self.screen_height - loc / float(self.scale))

	def draw_grids(self):
		
		for i in range(0, self.num_cars):
			pygame.draw.line(self.screen, self.color_white, (0, i * self.car_length_with_min_distance_pixels), (self.lanes * self.lane_width, i * self.car_length_with_min_distance_pixels) )

	def draw_car(self, color, pos):
		pygame.draw.rect(self.screen, color, pygame.Rect(pos[0], pos[1], 30, 30))

	def draw_graphics(self, action, reward):

		# Draw lanes
		self.screen.fill(self.color_green)
		
		for i in range(0, self.lanes):
			pygame.draw.line(self.screen, self.color_black, (i * self.lane_width, 0), (i * self.lane_width, self.screen_height), 3)
			pygame.draw.rect(self.screen, self.road_color, pygame.Rect((i * self.lane_width) + 3, 0, self.lane_width, self.screen_height))
			pygame.draw.line(self.screen, self.color_black, ((i * self.lane_width) + self.lane_width, 0), ((i * self.lane_width) + self.lane_width, self.screen_height), 3)
		
		self.draw_occupancy()

		# Blit cars
		#self.screen.blit(self.car_image, ((self.agent_lane * self.lane_width) + self.lane_width/2 - self.car_width/2 , self.agent_loc_pixels) )
		self.draw_car(self.color_red,((self.agent_lane * self.lane_width) + self.lane_width/2 - self.car_width/2 , self.agent_loc_pixels))

		for i  in range(0, self.lanes):
			for vehicle in self.vehicles_list[i]:
				loc = vehicle["loc"]
				lane = vehicle["lane"]
				#self.screen.blit(self.other_car_image, ((lane * self.lane_width) + self.lane_width/2 - self.car_width/2, loc))		
				self.draw_car(self.color_dark_green, ((lane * self.lane_width) + self.lane_width/2 - self.car_width/2, loc))
		
		

		# Display Information
		speed_string = 'Agent Speed : %.2f km/hr'%(self.agent_speed * 3.6)
		speed_text = self.font.render(speed_string, False, self.color_black)
		self.screen.blit(speed_text, ((self.lanes * self.lane_width) + 10, 10))

		max_speed_string = 'Max. Speed : %.2f km/hr'%(self.max_velocity * 3.6)
		max_speed_text = self.font.render(max_speed_string, False, self.color_black)
		self.screen.blit(max_speed_text, ((self.lanes * self.lane_width) + 10, 30))

		action_string = 'Action : %s'%(self.action_map[action])
		action_text = self.font.render(action_string, False, self.color_black)
		self.screen.blit(action_text, ((self.lanes * self.lane_width) + 10, 50))

		action_string = 'Time Elapsed : %d s'%(self.time_elapsed)
		action_text = self.font.render(action_string, False, self.color_black)
		self.screen.blit(action_text, ((self.lanes * self.lane_width) + 10, 70))

		action_string = 'Episode Density : %.1f'%(self.episode_density)
		action_text = self.font.render(action_string, False, self.color_black)
		self.screen.blit(action_text, ((self.lanes * self.lane_width) + 10, 90))

		action_string = 'Sampling Rate : %d Hz'%(1/self.time_period)
		action_text = self.font.render(action_string, False, self.color_black)
		self.screen.blit(action_text, ((self.lanes * self.lane_width) + 10, 110))

		action_string = 'Reward : %.2f'%(reward)
		action_text = self.font.render(action_string, False, self.color_black)
		self.screen.blit(action_text, ((self.lanes * self.lane_width) + 10, 130))

		pygame.display.flip()
		self.clock.tick(self.fps)

	def draw_occupancy(self):

		total_lines = int(self.view_size / self.cell_size)
		view_size_pixels = self.view_size_pixels / total_lines

		#print("Agent loc : ", self.agent_loc_pixels)
		#print("Start loc : ", self.agent_loc_pixels - self.view_size_pixels)

		for i in range(0, total_lines+1):
			pygame.draw.line(self.screen, self.color_white, (0, self.low + (i * view_size_pixels)), (self.lanes * self.lane_width, self.low + (i * view_size_pixels)))
			#print("val : ", self.low + (i * view_size_pixels))

		for i in range(0, self.occpancy_grid.shape[0]):
			for j in range(0, self.occpancy_grid.shape[1]):

				if self.occpancy_grid[i,j] == 1:
					pygame.draw.rect(self.screen, self.color_black, pygame.Rect(i*self.lane_width, self.low + (j * view_size_pixels), self.lane_width, self.view_size_pixels/self.view_cells))

	def init_occupancy(self):

		self.view_size_pixels = self.view_size * self.scale
		self.low = self.agent_loc_pixels - self.view_size_pixels
		self.view_cells = int(self.view_size/self.cell_size) # No of cells in view size
		self.cell_size_pixels = self.cell_size * self.scale
		self.car_cell_pixels = int(self.car_length_pixel/self.cell_size_pixels) + 1

	def get_occupancy(self):
		
		#print(self.occpancy_grid.shape)

		for i in range(0, self.lanes):
			for j in range(0, len(self.vehicles_list[i])):
				#print(self.vehicles_list[i][j])

				vehicle_end_pos = self.vehicles_list[i][j]['loc'] + self.car_length_pixel

				if vehicle_end_pos >= self.low and self.vehicles_list[i][j]['loc'] <= self.agent_loc_pixels:

					if vehicle_end_pos < self.agent_loc_pixels:

						#cell = (self.view_size_pixels - vehicle_end_pos) / self.cell_size_pixels
						cell = (self.agent_loc_pixels - vehicle_end_pos) / self.cell_size_pixels
						cell = math.floor(cell)

						if cell == self.occpancy_grid[i].shape[0]:
							pass
						else:
							self.occpancy_grid[i, cell] = 1
							self.occpancy_grid[i, cell+1:cell+self.car_cell_pixels] = 1

					else:

						cell = (self.agent_loc_pixels - self.vehicles_list[i][j]['loc']) / self.cell_size_pixels
						cell = math.floor(cell)
						#self.occpancy_grid[i, cell] = 1
						self.occpancy_grid[i, 0:cell+1] = 1

				else:
					pass
					#print("Not In Area")

		for i in range(0, self.lanes):
			self.occpancy_grid[i] = np.flip(self.occpancy_grid[i])

	def get_state(self):

		return np.concatenate((self.occpancy_grid.flatten(), np.array(self.agent_speed).reshape(1,)/self.max_velocity, np.array(self.agent_lane).reshape(1,)/(self.lanes-1))).copy() 

	def reset(self, density=None):


		self.episode_done = False

		if density == None:
			density = self.densities[np.random.randint(0, self.densities.shape[0])]

		if self.config["use-density"]:
			density = self.config["density"]

		self.episode_density = density
		self.time_elapsed = 0.0
		self.agent_lane = np.random.randint(0, self.lanes)
		if self.config["init-velocity"]:
			self.agent_speed = self.config["init-vel"]
		else:
			self.agent_speed = np.random.uniform(low=0.0, high=self.max_velocity)
		#print("Agent Start Speed : ", self.agent_speed)
		self.occpancy_grid = np.zeros((self.lanes, int(self.view_cells)))

		self.vehicles_list = {}

		for i in range(0, self.lanes):
			self.vehicles_list[i] = []

		tmp = np.random.rand(self.lanes, len(self.init_pos_list))

		for i in range(0, self.lanes):
			for j in range(0, tmp.shape[1]):

				if not (i,j) == (self.agent_lane, len(self.init_pos_list)-2):
					if tmp[i,j] <= density:
						tmp_vehicle_dict = {}
						tmp_vehicle_dict["loc"] = self.init_pos_list[j]
						tmp_vehicle_dict["lane"] = i
						self.vehicles_list[i].append(copy.deepcopy(tmp_vehicle_dict))

		self.screen_tracker = 0.0
		self.get_occupancy()

		if self.render:
			self.draw_graphics(None, 0.0)

		return self.get_state()

	def distance_travelled_metre(self, u, t, a):
		res = u * t + (0.5 * a * t * t)
		return res

	def new_velocity(self, u, t, a):
		res = u + (a * t)
		return res

	def check_collision(self):

		collision = False

		if len(self.vehicles_list[self.agent_lane]) > 0:
			agent_lane_cars = copy.deepcopy(self.vehicles_list[self.agent_lane])
			agent_lane_cars.append({"loc": self.agent_loc_pixels})
			agent_lane_cars = sorted(agent_lane_cars, key=itemgetter('loc'))
			
			for i in range(0, len(agent_lane_cars)):
				if agent_lane_cars[i]['loc'] == self.agent_loc_pixels:
					agent_index = i
					break


			if agent_index == 0:
				collision = False
			else:

				prev_car_loc = agent_lane_cars[agent_index-1]["loc"]

				gap = self.agent_loc_pixels - (prev_car_loc + self.car_length_pixel)
				if gap < 1.0:
					collision = True

		else:
			collision = False

		return collision

	def check_collision_v2(self):

		if self.occpancy_grid[self.agent_lane, -1] == 1:
			return True
		else:
			return False
		#print(self.occpancy_grid[self.agent_lane, -1])

	def step(self, action):
		
		generate_car = False

		self.time_elapsed += self.time_period
		reward = 0.0

		if action == 0:

			acc = 0.0
			old_speed = self.agent_speed

			if self.agent_speed > self.max_velocity:
				self.agent_speed = self.new_velocity(self.agent_speed, self.time_period, 0)
				acc = 0.0
			else:
				self.agent_speed = self.new_velocity(self.agent_speed, self.time_period, self.a)
				acc = self.a

			done = self.check_collision_v2()

			if done:
				travelled = 0.0
				self.agent_speed = 0.0
				travelled_pixels = self.scale * travelled
			else:
				travelled = self.distance_travelled_metre(self.agent_speed, self.time_period, acc)
				travelled_pixels = self.scale * travelled
				#print(travelled_pixels)

		if action == 1:

			dec = 0.0
			old_speed = self.agent_speed

			if self.agent_speed <= 0.0:
				self.agent_speed = 0
				dec = 0.0
			else:
				self.agent_speed = self.new_velocity(self.agent_speed, self.time_period, -self.b)
				dec = -self.b

				if self.agent_speed < 0.0:
					self.agent_speed = 0.0

			done = self.check_collision_v2()

			if done:
				travelled = 0.0
				self.agent_speed = 0.0
				travelled_pixels = self.scale * travelled
			else:
				travelled = self.distance_travelled_metre(self.agent_speed, self.time_period, dec)
				if travelled < 0.0:
					travelled = 0.0
				travelled_pixels = self.scale * travelled

		if action == 2:

			acc = 0.0
			self.agent_speed = self.new_velocity(self.agent_speed, self.time_period, acc)

			done = self.check_collision_v2()

			if done:
				travelled = 0.0
				self.agent_speed = 0.0
				travelled_pixels = self.scale * travelled
			else:
				travelled = self.distance_travelled_metre(self.agent_speed, self.time_period, acc)
				travelled_pixels = self.scale * travelled


		# Remove Cars
		for i in range(0, self.lanes):
			for j in range(0, len(self.vehicles_list[i])):

				self.vehicles_list[i][j]["loc"] += travelled_pixels

				if self.vehicles_list[i][j]["loc"] > self.screen_height:
					self.vehicles_list[i].pop(j)
					break

		self.screen_tracker += travelled_pixels

		if self.screen_tracker > (self.init_pos_list[1] - self.init_pos_list[0]):
			generate_car = True
			self.screen_tracker = 0.0
		
		# Add Cars
		if generate_car:

			tmp = np.random.rand(self.lanes)

			for i in range(0, self.lanes):
				if tmp[i] <= self.episode_density:
					tmp_vehicle_dict = {}
					tmp_vehicle_dict["loc"] = self.init_pos_list[0]
					tmp_vehicle_dict["lane"] = i
					self.vehicles_list[i].append(copy.deepcopy(tmp_vehicle_dict))

		self.occpancy_grid = np.zeros((self.lanes, int(self.view_cells)))
		self.get_occupancy()

		done = self.check_collision_v2()

		#done = self.check_collision()

		#print("Occupancy : ")
		#print(self.occpancy_grid)
		#print("State : ", self.get_state())

		if done:
			reward = -10
			self.info_dict["collide_count"] += 1
			self.agent_speed = 0.0
			#self.episode_done = True
		else:
			reward = self.agent_speed / self.max_velocity

		if self.render:
			self.draw_graphics(action, reward)

		return (self.get_state(), reward, done, self.info_dict)

if __name__ ==  "__main__":

	with open("sim-config.yaml", "r") as handle:
		sim_config = yaml.load(handle)

	obj = sim(sim_config)
	state = obj.reset(0.1)
	
	#print(obj.observation_space)
	#print(obj.action_space)
	#print(state)
	#time.sleep(100)
	count = 0

	while True:

		state, reward, done, _ = obj.step(np.random.randint(0, 3))
		#state, reward, done, _ = obj.step(0)
		print(state)
		#print(state)
		#print(done)
		'''
		print("------------------------")
		print("Agent Lane : ", obj.agent_lane)
		print(state[0:30].reshape(3, 10))
		print("------------------------")
		'''

		if done:
			break