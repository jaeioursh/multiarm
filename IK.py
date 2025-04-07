import mujoco
import numpy as np


class IK:

    def __init__(self):
        # Integration timestep in seconds. This corresponds to the amount of time the joint
        # velocities will be integrated for to obtain the desired joint positions.
        self.integration_dt = 1.0

        # Damping term for the pseudoinverse. This is used to prevent joint velocities from
        # becoming too large when the Jacobian is close to singular.
        self.damping = 1e-4

        # Whether to enable gravity compensation.
        self.gravity_compensation = True

        # Simulation timestep in seconds.
        # self.dt = 0.002

        # Maximum allowable joint velocity in rad/s. Set to 0 to disable.
        self.max_angvel = 3.0

    def set_targets(self, poss, quats):
        self.target_quat = quats
        self.target_pos = poss

    def get_q(self, model, data, n_agents):

        # Override the simulation timestep.
        # model.opt.timestep = dt
        prefixes = ["A_", "B_", "C_", "D_"]
        # End-effector site we wish to control, in this case a site attached to the last
        # link (wrist_3_link) of the robot.
        dqs = np.zeros(model.nv)
        for agent_idx in range(n_agents):
            prefix = prefixes[agent_idx]
            site_id = model.site(prefix+"attachment_site").id

            # Name of bodies we wish to apply gravity compensation to.
            body_names = [
                "shoulder_link",
                "upper_arm_link",
                "forearm_link",
                "wrist_1_link",
                "wrist_2_link",
                "wrist_3_link",
            ]
            body_names = [prefix + bn for bn in body_names]
            body_ids = [model.body(name).id for name in body_names]
            if self.gravity_compensation:
                model.body_gravcomp[body_ids] = 1.0
            true_nv = model.nv//n_agents
            # Pre-allocate numpy arrays.
            jac = np.zeros((6, model.nv))
            diag = self.damping * np.eye(6)
            error = np.zeros(6)
            error_pos = error[:3]
            error_ori = error[3:]
            site_quat = np.zeros(4)
            site_quat_conj = np.zeros(4)
            error_quat = np.zeros(4)

            # Position error.
            error_pos[:] = self.target_pos[agent_idx] - data.site(site_id).xpos

            # Orientation error.
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(
                error_quat, self.target_quat[agent_idx], site_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

            # Get the Jacobian with respect to the end-effector site.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
            # Solve system of equations: J @ dq = error.
            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

            # Scale down joint velocities if they exceed maximum.
            if self.max_angvel > 0:
                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > self.max_angvel:
                    dq *= self.max_angvel / dq_abs_max
            dqs += dq
            # Integrate joint velocities to obtain joint positions.
        dq = np.array(dqs)
        q = data.qpos.copy()
        mujoco.mj_integratePos(model, q, dq, self.integration_dt)
        q = q[7:]
        # Set the control signal.
        np.clip(q, *model.actuator_ctrlrange.T, out=q)
        lower, upper = np.array(model.actuator_ctrlrange).T
        q /= upper
        return q
