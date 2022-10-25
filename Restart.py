import taichi as ti 
import numpy as np

def ParticleReload(PRINT):
	data = np.load("MonitorDEM{0:06d}.npz".format(PRINT))
	t_start = data["t_current"]
	particle_num = data["particleNum"]

	pos_x = data["pos_x"]
	pos_y = data["pos_y"]
	pos_z = data["pos_z"]

	vel_x = data["vel_x"]
	vel_y = data["vel_y"]
	vel_z = data["vel_z"]
	
	w_x = data["w_x"]
	w_y = data["w_y"]
	w_z = data["w_z"]
