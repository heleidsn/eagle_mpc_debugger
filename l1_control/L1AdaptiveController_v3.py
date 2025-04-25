'''
Author: Lei He
Date: 2024-09-07 17:26:46
LastEditTime: 2025-04-21 19:09:14
Description: L1 adaptive controller for systems modelled with Pinocchio.
             Supports full-state or velocity-only adaptation/prediction.
Version 4: Refactored for clarity and conciseness.
Github: https://github.com/heleidsn
'''

import numpy as np
from numpy import linalg as LA
from scipy import linalg as sLA
from scipy.spatial.transform import Rotation as R
import pinocchio as pin

class L1AdaptiveControllerRefactored:
    """
    L1 Adaptive Controller augmenting a baseline controller (e.g., MPC).

    Handles systems modeled by Pinocchio, potentially with robotic arms.
    Supports different configurations via flags.

    Args:
        dt (float): Controller time step.
        robot_model (pin.Model): Pinocchio robot model.
        As_coef (float): Coefficient for the diagonal Hurwitz matrix A_s (e.g., -100).
        filter_time_constant (float): Time constant for the low-pass filter on adaptive signals.
        using_full_state (bool, optional): If True, use full state for adaptation/prediction.
                                           If False, use velocity-only. Defaults to False.
        flag_using_z_ref (bool, optional): Flag to use reference state error in predictor/tracking. Defaults to False.
        using_vel_disturbance (bool, optional): Flag to add disturbance compensation to velocity prediction dynamics. Defaults to False.
    """
    def __init__(self, dt, robot_model, As_coef, filter_time_constant,
                 using_full_state=False, flag_using_z_ref=False, using_vel_disturbance=False):

        self.dt = dt
        self.robot_model = robot_model
        self.robot_model_data = self.robot_model.createData()

        # --- Configuration Flags ---
        self.using_full_state = using_full_state
        self.flag_using_z_ref = flag_using_z_ref
        self.using_vel_disturbance = using_vel_disturbance # Controls velocity term addition in z_hat update

        # --- Dimensions ---
        self.nq = self.robot_model.nq  # Configuration space dim (including quaternion if applicable)
        self.nv = self.robot_model.nv  # Velocity space / tangent space dim / control dim
        self.state_dim = self.nq + self.nv  # Dimension of state used in Pinocchio (quat representation)
        # Assuming state_dim_euler is used for internal z_hat, z_real, z_ref, sig_hat, A_s
        # Typically: base_pos(3) + base_orient_euler(3) + arm_joints(nq-7) + base_vel(6) + arm_vel(nv-6)
        # Or if nv includes base vel (6) + arm vel (nq-7): state_dim_euler = 3 + 3 + (nq-7) + nv = nq - 1 + nv
        self.state_dim_euler = self.nq - 1 + self.nv if self.nq >= 7 else 3 + 3 + self.nv # Simplified assumption
        self.control_dim = self.nv # Control input dimension often equals velocity dimension

        self.use_arm = self.nq > 7
        if self.use_arm:
            self.arm_joint_num = self.nq - 7
            self.filter_num = 3 # Force, Torque, Arm Torque
        else:
            self.arm_joint_num = 0
            self.filter_num = 2 # Force, Torque

        # --- L1 Parameters ---
        # Hurwitz Matrix A_s (diagonal)
        self.A_s = np.diag(np.full(self.state_dim_euler, As_coef))
        self.expm_A_s_dt = sLA.expm(self.A_s * self.dt)

        # Low-pass filter time constants
        self.filter_time_const = np.full(self.filter_num, filter_time_constant)

        # --- Optional Tracking Controller Parameters ---
        # TODO: Make these configurable if needed
        self.tracking_error_p_gain = np.zeros(self.nv)
        if self.nv >= 6: # Basic default gains, adjust as needed
             # Gains for base Euler angles (assuming order roll, pitch, yaw) and arm joints
             base_orient_gain_indices = slice(3, 6) # Indices for base angular velocity control dim
             arm_joint_gain_indices = slice(6, self.nv)
             # Example gains (tune these!)
             # self.tracking_error_p_gain[base_orient_gain_indices] = np.array([20, 20, 3]) # Roll, Pitch, Yaw gain
             # if self.use_arm:
             #     self.tracking_error_p_gain[arm_joint_gain_indices] = 0.5 # Example arm gain
        self.tracking_error_v_gain = np.zeros(self.nv) # Example velocity gain

        # --- Initialize State Variables ---
        self.init_controller()

    def init_controller(self):
        """Initializes controller states and estimates."""
        # Pinocchio full state (pos, vel) - Uses Quaternion representation
        self.current_state = np.zeros(self.state_dim)
        self.z_ref_all = np.zeros(self.state_dim) # Reference state in Pinocchio format

        # Internal states using Euler representation for orientation
        self.z_hat = np.zeros(self.state_dim_euler) # Estimated state (Euler)
        self.z_ref = np.zeros(self.state_dim_euler) # Reference state (Euler)
        self.z_real = np.zeros(self.state_dim_euler) # Measured state (Euler)

        # L1 specific states
        self.sig_hat = np.zeros(self.state_dim_euler) # Estimated disturbance/uncertainty (Euler state dim)
        self.z_tilde = np.zeros(self.state_dim_euler) # State estimation error z_hat - z_real
        self.z_tilde_ref = np.zeros(self.state_dim_euler) # State predictor error z_hat - z_ref
        self.z_tilde_tracking = np.zeros(self.state_dim_euler) # Tracking error z_real - z_ref

        # Control inputs
        self.u_ad = np.zeros(self.control_dim) # Adaptive control
        self.u_mpc = np.zeros(self.control_dim) # Baseline control (e.g., MPC)
        self.u_tracking = np.zeros(self.control_dim) # Optional tracking feedback

        # Previous filter states (size matches control_dim)
        self.sig_filter_prev = np.zeros(self.control_dim)

        # Separate tracking errors (consider if calculation can be simplified)
        # These might be redundant if z_tilde_tracking is sufficient
        self.tracking_error_pos = np.zeros(self.nq -1 if self.nq >=7 else self.nq) # Pos + Euler + Arm Pos
        self.tracking_error_vel = np.zeros(self.nv)

    # --- State Conversion Utilities ---

    def _split_state_quat(self, state_vec):
        """Splits state vector (quat representation) into q and v."""
        q = state_vec[:self.nq]
        v = state_vec[self.nq:]
        return q, v

    def _split_state_euler(self, state_vec_euler):
        """Splits Euler state vector into pos, euler, arm_pos, lin_vel, ang_vel, arm_vel."""
        pos = state_vec_euler[0:3]
        euler = state_vec_euler[3:6]
        idx_after_euler = 6
        arm_pos = np.array([])
        if self.use_arm:
            arm_pos = state_vec_euler[idx_after_euler : idx_after_euler + self.arm_joint_num]
            idx_after_euler += self.arm_joint_num

        lin_vel = state_vec_euler[idx_after_euler : idx_after_euler + 3] # Assumes lin vel is next
        ang_vel = state_vec_euler[idx_after_euler + 3 : idx_after_euler + 6] # Assumes ang vel follows
        idx_after_ang_vel = idx_after_euler + 6
        arm_vel = np.array([])
        if self.use_arm:
             arm_vel = state_vec_euler[idx_after_ang_vel : idx_after_ang_vel + self.arm_joint_num] # Assumes arm vel is last

        return pos, euler, arm_pos, lin_vel, ang_vel, arm_vel

    def _state_quat_to_euler(self, state_quat_vec):
        """Converts full state vector from quat to euler representation (angles in rad)."""
        q, v = self._split_state_quat(state_quat_vec)
        pos = q[:3]
        quat = q[3:7]
        arm_q = q[7:] # Empty if no arm

        rotation = R.from_quat(quat) # Expects [x, y, z, w] ? Check Pinocchio convention
        euler_angles = rotation.as_euler('xyz', degrees=False)

        return np.concatenate((pos, euler_angles, arm_q, v))

    def _state_euler_to_quat(self, state_euler_vec):
        """Converts full state vector from euler to quat representation."""
        pos, euler, arm_pos, lin_vel, ang_vel, arm_vel = self._split_state_euler(state_euler_vec)

        rotation = R.from_euler('xyz', euler, degrees=False)
        quat = rotation.as_quat() # Check order [x, y, z, w]?

        q = np.concatenate((pos, quat, arm_pos))
        v = np.concatenate((lin_vel, ang_vel, arm_vel)) # Assumes v order matches
        return np.concatenate((q, v))

    # --- Core L1 Update Steps ---

    def update_measurements(self, current_state_quat, z_ref_quat=None):
         """Updates internal states based on new measurements and references."""
         self.current_state = np.asarray(current_state_quat).copy()
         self.z_real = self._state_quat_to_euler(self.current_state)

         if z_ref_quat is not None:
            self.z_ref_all = np.asarray(z_ref_quat).copy()
            self.z_ref = self._state_quat_to_euler(self.z_ref_all)
         # If z_ref_quat is None, self.z_ref retains its previous value or initial zeros

         self._update_errors() # Calculate errors based on updated states

    def _update_errors(self):
        """Calculates various error terms."""
        # State prediction error
        self.z_tilde = self.z_hat - self.z_real
        if not self.using_full_state:
            # If only using velocity, zero out non-velocity parts of z_tilde
            # Assuming velocity states are the last nv elements in the Euler state
            self.z_tilde[:-self.nv] = 0.0

        # Predictor-reference error
        self.z_tilde_ref = self.z_hat - self.z_ref

        # Tracking error (Euler representation)
        self.z_tilde_tracking = self.z_real - self.z_ref

        # --- Calculate specific tracking errors for optional controller ---
        # Position error
        pos_real, euler_real, arm_pos_real, _, _, _ = self._split_state_euler(self.z_real)
        pos_ref, euler_ref, arm_pos_ref, _, _, _ = self._split_state_euler(self.z_ref)
        pos_err = pos_real - pos_ref
        # Orientation error (using Euler difference - careful with wrap-around)
        # Consider using rotation matrix/quaternion difference for better orientation error metric
        orient_err = euler_real - euler_ref # Simple difference, wrap pi/-pi if needed
        arm_pos_err = arm_pos_real - arm_pos_ref
        self.tracking_error_pos = np.concatenate((pos_err, orient_err, arm_pos_err))

        # Velocity error
        self.tracking_error_vel = self.z_real[-self.nv:] - self.z_ref[-self.nv:]


    def _update_predictor(self, u_baseline):
        """Updates the state predictor z_hat."""
        self.u_mpc = np.asarray(u_baseline).copy() # Store baseline control

        if self.using_full_state:
            self._update_z_hat_full_state()
        else:
            self._update_z_hat_velocity_only()

    def _update_z_hat_velocity_only(self):
        """Predictor update using only velocity error compensation (Simpler L1 variant)."""
        z_hat_prev = self.z_hat.copy()
        q_meas, v_meas = self._split_state_quat(self.current_state)

        # --- Calculate Control for Dynamics ---
        # NOTE: Review required: Standard L1 predictor uses u_baseline + sig_hat, not u_baseline + u_ad + sig_hat.
        # Assuming sig_hat relevant part corresponds to velocity states (last nv elements of sig_hat)
        # Assuming u_ad corresponds to velocity control dimension (nv)
        # Assuming sig_hat estimation provides equivalent input disturbance
        sig_hat_eff = self.sig_hat[-self.nv:] # Effective disturbance estimate for velocity dynamics
        u_pred = self.u_mpc + self.u_ad + sig_hat_eff
        # --- End NOTE ---

        # --- Calculate Predictor Derivative ---
        # Calculate acceleration based on *measured* state and predictor control input
        # NOTE: Using measured state (q_meas, v_meas) instead of predicted (z_hat_prev)
        a_nominal = pin.aba(self.robot_model, self.robot_model_data, q_meas, v_meas, u_pred)

        # Calculate full correction term
        # NOTE: Using z_tilde (potentially with pos part zeroed) or z_tilde_ref based on flag
        z_err = self.z_tilde_ref if self.flag_using_z_ref else self.z_tilde
        correction_term = self.A_s @ z_err

        # Derivative estimate for velocity part only
        # NOTE: Standard L1 would apply correction_term to full state derivative
        z_hat_dot_vel = a_nominal + correction_term[-self.nv:] # Applying only velocity part of correction

        # --- Euler Integration ---
        z_hat_new = z_hat_prev.copy()
        # Update only the velocity part of z_hat (last nv elements)
        z_hat_new[-self.nv:] = z_hat_prev[-self.nv:] + z_hat_dot_vel * self.dt
        # Position part remains unchanged in this velocity-only predictor update
        # z_hat_new[:-self.nv] = z_hat_prev[:-self.nv] # Explicitly keep position part same

        self.z_hat = z_hat_new

    def _update_z_hat_full_state(self):
        """Predictor update using full state error compensation."""
        # Based on original update_z_hat_all, needs review for L1 correctness

        # --- Get Previous/Current States ---
        z_hat_prev_euler = self.z_hat.copy()
        q_meas, v_meas = self._split_state_quat(self.current_state)

        # --- Calculate Control for Dynamics ---
        # NOTE: Review required: Standard L1 predictor uses u_baseline + sig_hat, not u_baseline + u_ad + sig_hat.
        # Assuming sig_hat relevant part corresponds to velocity states (last nv elements of sig_hat)
        sig_hat_eff = self.sig_hat[-self.nv:]
        u_pred = self.u_mpc + self.u_ad + sig_hat_eff # Using velocity part of sig_hat here
        # --- End NOTE ---

        # --- Calculate Correction Term ---
        z_err = self.z_tilde_ref if self.flag_using_z_ref else self.z_tilde
        correction_term = self.A_s @ z_err # Full state correction term

        # --- Calculate State Derivative ---
        # NOTE: This part deviates significantly from standard L1 predictor structure.
        # Original code added correction term parts at different stages.
        # Refactored approach: Calculate nominal derivatives + full correction term.
        # Using measured state for nominal dynamics calculation:
        a_nominal = pin.aba(self.robot_model, self.robot_model_data, q_meas, v_meas, u_pred)
        v_nominal = v_meas # Approximation: velocity derivative part = measured velocity

        # Combine into Euler state derivative (Pos_dot, Euler_dot, ArmPos_dot, LinVel_dot, AngVel_dot, ArmVel_dot)
        # Requires careful mapping from v_nominal (body vel) and a_nominal (body acc) to Euler state derivatives
        # This mapping is complex (involves orientation jacobians, etc.) and was approximated in the original code.
        # For simplicity, we mimic the *effect* of the original code's Euler integration approach here,
        # but this section likely needs a more rigorous derivation based on chosen state representation.

        # Mimicking original Euler Integration logic (split update):
        # 1. Predict next velocity based on acceleration + partial correction
        a_corrected_vel = a_nominal + correction_term[-self.nv:] # Add vel part of correction to accel
        v_hat_next = z_hat_prev_euler[-self.nv:] + a_corrected_vel * self.dt # Integrate accel

        # 2. Add optional velocity disturbance correction
        if self.using_vel_disturbance:
             # Add pos/angle part of correction directly to *integrated* velocity? Seems odd.
             # Original code added A_mul_z_tilde[:nv] or similar. Let's use correction_term[:-nv]
             # This step is highly suspect and likely needs theoretical revision.
             # Assuming correction_term maps pos/angle error to velocity correction:
             v_hat_next += correction_term[:-self.nv] * self.dt # Tentative, needs verification

        # 3. Integrate position/orientation using *predicted* next velocity
        # Convert previous Euler state estimate back to Quat for integration
        z_hat_prev_quat_vec = self._state_euler_to_quat(z_hat_prev_euler)
        q_hat_prev, _ = self._split_state_quat(z_hat_prev_quat_vec) # Only need q part
        pos_hat_prev = q_hat_prev[:3]
        quat_hat_prev = q_hat_prev[3:7]
        arm_q_hat_prev = q_hat_prev[7:]

        # Get components of predicted next velocity
        lin_vel_hat_next = v_hat_next[:3]
        ang_vel_hat_next = v_hat_next[3:6]
        arm_vel_hat_next = v_hat_next[6:]

        # Integrate orientation
        rotation_prev = R.from_quat(quat_hat_prev)
        delta_rotation = R.from_rotvec(ang_vel_hat_next[:3] * self.dt) # Use base angular velocity
        rotation_next = rotation_prev * delta_rotation
        quat_hat_next = rotation_next.as_quat()

        # Integrate position (convert body linear velocity to world frame)
        lin_vel_world_next = rotation_prev.apply(lin_vel_hat_next) # Use previous orientation for transform
        pos_hat_next = pos_hat_prev + lin_vel_world_next * self.dt

        # Integrate arm position
        arm_q_hat_next = arm_q_hat_prev + arm_vel_hat_next * self.dt # Assuming direct integration

        # 4. Combine into new Euler state estimate
        self.z_hat = np.concatenate((pos_hat_next, rotation_next.as_euler('xyz', degrees=False), arm_q_hat_next, v_hat_next))


    def _update_adaptation_law(self):
        """Updates the disturbance/uncertainty estimate sig_hat."""
        if self.using_full_state:
            self._update_sig_hat_full_state()
        else:
            self._update_sig_hat_velocity_only()

    def _calculate_common_adaptation_terms(self):
        """Calculates terms common to both adaptation laws."""
        # PHI matrix for piece-wise constant law
        # Ensure A_s is invertible (true if Hurwitz and no zero eigenvalues)
        try:
            A_s_inv = LA.inv(self.A_s)
        except LA.LinAlgError:
            print("Error: A_s matrix is singular, cannot compute adaptation law.")
            return None, None # Indicate error
        PHI = A_s_inv @ (self.expm_A_s_dt - np.identity(self.state_dim_euler))

        # Ensure PHI is invertible
        try:
            PHI_inv = LA.inv(PHI)
        except LA.LinAlgError:
             print("Error: PHI matrix is singular, cannot compute adaptation law.")
             # This might happen if dt is too large relative to A_s eigenvalues
             return None, None # Indicate error

        mu = self.expm_A_s_dt @ self.z_tilde # Use z_tilde = z_hat - z_real
        PHI_inv_mul_mu = PHI_inv @ mu

        return PHI_inv_mul_mu, True # Return common term and success flag

    def _update_sig_hat_velocity_only(self):
        """Adaptation law using only velocity state error info implicitly."""
        # Assumes z_tilde position part is zeroed if not using_full_state
        common_term, success = self._calculate_common_adaptation_terms()
        if not success:
            # Keep previous estimate if calculation failed
            print("Warning: Adaptation law calculation failed (vel only). Keeping previous sig_hat.")
            return

        # Get inertia matrix (only needed for velocity part)
        q_meas, _ = self._split_state_quat(self.current_state)
        M = pin.crba(self.robot_model, self.robot_model_data, q_meas) # Only velocity part matters

        # B_bar_inv equivalent for velocity dynamics is M
        B_bar_inv_vel = M

        # Calculate sigma_hat for velocity part
        # Assumes common_term[-nv:] corresponds to velocity error dynamics influence
        # NOTE: Review Required - Is multiplying by M correct here?
        # The formula sigma = -B_inv * PHI_inv * mu assumes a specific state structure
        # and input matrix B. Without knowing the exact derivation used, assume the original code's
        # intention was that M relates the velocity part of the error dynamics result
        # back to the input disturbance affecting velocity.
        sigma_hat_vel = -B_bar_inv_vel @ common_term[-self.nv:]

        # Update only the velocity part of sig_hat
        self.sig_hat[-self.nv:] = sigma_hat_vel
        # Keep position part zero or unchanged (depending on desired behavior)
        self.sig_hat[:-self.nv] = 0.0 # Zero out position disturbance estimate


    def _update_sig_hat_full_state(self):
        """Adaptation law using full state error information."""
        common_term, success = self._calculate_common_adaptation_terms()
        if not success:
            print("Warning: Adaptation law calculation failed (full state). Keeping previous sig_hat.")
            return

        # Get augmented inertia matrix M_aug ~ B_bar_inv
        q_meas, _ = self._split_state_quat(self.current_state)
        M = pin.crba(self.robot_model, self.robot_model_data, q_meas)
        M_aug = np.zeros((self.state_dim_euler, self.state_dim_euler))
        # Assuming upper part corresponds to position/orientation state (nv - 1 elements?)
        # Assuming lower part corresponds to velocity state (nv elements)
        # This structure needs careful verification based on state definition and L1 derivation
        pos_orient_dim = self.state_dim_euler - self.nv
        M_aug[:pos_orient_dim, :pos_orient_dim] = np.identity(pos_orient_dim) # Assume direct mapping for pos/orient "disturbance"
        M_aug[-self.nv:, -self.nv:] = M # Use inertia for velocity part

        B_bar_inv = M_aug

        # Calculate full sigma_hat
        sigma_hat_full = -B_bar_inv @ common_term
        self.sig_hat = sigma_hat_full

    def _low_pass_filter(self, time_const, current_val, prev_filtered_val):
        """Applies a first-order low-pass filter."""
        if time_const <= 0: # Avoid division by zero or unstable filter
            return current_val
        alpha = self.dt / (self.dt + time_const)
        return (1.0 - alpha) * prev_filtered_val + alpha * current_val

    def update_adaptive_control(self, u_limits_min=None, u_limits_max=None):
        """Calculates the adaptive control signal u_ad based on filtered sig_hat."""
        # Filter the part of sig_hat corresponding to control inputs (velocity dynamics)
        sig_hat_to_filter = self.sig_hat[-self.control_dim:]

        # Apply filter(s)
        # Assuming filter_time_const has entries for force, torque, arm torque if applicable
        filtered_signal = np.zeros(self.control_dim)
        start_idx = 0

        # Filter force part (first 3 controls)
        if self.control_dim >= 3:
            filtered_signal[start_idx:start_idx+3] = self._low_pass_filter(
                self.filter_time_const[0],
                sig_hat_to_filter[start_idx:start_idx+3],
                self.sig_filter_prev[start_idx:start_idx+3]
            )
            start_idx += 3

        # Filter torque part (next 3 controls)
        if self.control_dim >= 6:
            filtered_signal[start_idx:start_idx+3] = self._low_pass_filter(
                self.filter_time_const[1],
                sig_hat_to_filter[start_idx:start_idx+3],
                self.sig_filter_prev[start_idx:start_idx+3]
            )
            start_idx += 3

        # Filter arm part (remaining controls)
        if self.use_arm and self.control_dim > 6:
            filtered_signal[start_idx:] = self._low_pass_filter(
                self.filter_time_const[2],
                sig_hat_to_filter[start_idx:],
                self.sig_filter_prev[start_idx:]
            )

        # Store current filtered value for next iteration
        self.sig_filter_prev = filtered_signal.copy()

        # Calculate adaptive control signal
        self.u_ad = -filtered_signal

        # Apply limits if provided
        if u_limits_min is not None and u_limits_max is not None:
             self.u_ad = np.clip(self.u_ad, u_limits_min, u_limits_max)
        elif u_limits_min is not None: # Only min limit
             self.u_ad = np.maximum(self.u_ad, u_limits_min)
        elif u_limits_max is not None: # Only max limit
             self.u_ad = np.minimum(self.u_ad, u_limits_max)


        # --- Optional: Add Tracking Feedback ---
        self.u_tracking = np.zeros(self.control_dim)
        if self.flag_using_z_ref:
            # Simple P-controller based on tracking errors
            # NOTE: Using Euler angle differences for orientation can be problematic (wrap-around)
            # Consider using proper orientation error metrics.
            # Applying gains only to velocity state errors corresponding to control dims
             # Example: PD-like feedback on tracking error (using z_tilde_tracking)
             # pos_orient_err = self.z_tilde_tracking[:-self.nv] # Error in pos/euler/arm_pos
             vel_err = self.z_tilde_tracking[-self.nv:]  # Error in velocities

             # Map position/orientation error to control output using gains (simplified)
             # This mapping requires careful thought (e.g., Jacobian)
             # Simplified: use gains directly on corresponding velocity errors for now
             # Need a better mapping from self.tracking_error_pos to control_dim outputs
             # For now, only use velocity error feedback as an example:
             self.u_tracking = -self.tracking_error_v_gain * vel_err
             # A more complex term involving self.tracking_error_pos mapped to control_dim could be added here.


        return self.u_ad + self.u_tracking


    # --- Main Execution Method ---
    def compute_control(self, current_state_quat, u_baseline, z_ref_quat=None, u_limits=None):
        """
        Computes the total control signal (baseline + adaptive + tracking).

        Args:
            current_state_quat (np.ndarray): Current full state [q, v] with quaternion orientation.
            u_baseline (np.ndarray): Control signal from the baseline controller (e.g., MPC).
            z_ref_quat (np.ndarray, optional): Reference state [q_ref, v_ref] with quaternion. Defaults to None.
            u_limits (tuple, optional): Tuple of (min_limits_array, max_limits_array) for u_ad. Defaults to None.

        Returns:
            np.ndarray: The total control signal u_total = u_baseline + u_ad + u_tracking.
        """
        # 1. Update internal states based on measurements
        self.update_measurements(current_state_quat, z_ref_quat)

        # 2. Update state predictor (depends on u_ad from previous step, uses u_baseline now)
        # Note: Predictor uses u_ad from the *previous* cycle.
        self._update_predictor(u_baseline)

        # 3. Update adaptation law (estimates sig_hat based on current error z_tilde)
        self._update_adaptation_law()

        # 4. Calculate new adaptive control signal u_ad (and optional u_tracking)
        u_limits_min, u_limits_max = None, None
        if u_limits is not None and len(u_limits) == 2:
             u_limits_min, u_limits_max = u_limits
        u_adaptive_component = self.update_adaptive_control(u_limits_min, u_limits_max)

        # 5. Calculate total control
        u_total = u_baseline + u_adaptive_component
        self.u_mpc = u_baseline # Store for reference
        self.u_ad = u_adaptive_component # Store for reference

        return u_total