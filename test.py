import time

import mujoco
import mujoco.viewer
import numpy as np
m = mujoco.MjModel.from_xml_path('ur5e/multiscene.xml')
d = mujoco.MjData(m)

n_joints=m.nu
d.qpos=np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]*3)
with mujoco.viewer.launch_passive(m, d) as viewer:
	# Close the viewer automatically after 30 wall-seconds.
	start = time.time()
	viewer.cam.distance = m.stat.extent * 7.0
	while viewer.is_running() and time.time() - start < 30:
		step_start = time.time()
		state=d.qpos
		print(state)
		d.ctrl=np.zeros(n_joints)+1.0
		# mj_step can be replaced with code that also evaluates
		# a policy and applies a control signal before stepping the physics.
		mujoco.mj_step(m, d)

		# Pick up changes to the physics state, apply perturbations, update options from GUI.
		viewer.sync()

		# Rudimentary time keeping, will drift relative to wall clock.
		time_until_next_step = m.opt.timestep - (time.time() - step_start)
		if time_until_next_step > 0:
			time.sleep(time_until_next_step)