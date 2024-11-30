import time
import mujoco
import mujoco.viewer
import numpy as np
import queue

from dial_mpc.utils.io_utils import get_model_path

class PredictionViz:
    def __init__(self, predictions_path, states_path):
        self.predictions = np.load(predictions_path)
        self.states = np.load(states_path)

        # self.predictions = self.predictions[:100]
        # self.states = self.states[:100]
        
        # Constants from your data shape
        self.T = self.predictions.shape[0]  # 100 timesteps
        self.n_diffuse = self.predictions.shape[1]  # 10 diffusion steps
        self.H = self.predictions.shape[2] - 1  # 16 horizon steps
        self.nbody = self.predictions.shape[3]  # 13 bodies

        self.vis_body_ids = [3, 6, 9, 12]
        
        # Current visualization state
        self.current_t = 0
        
        # Initialize MuJoCo
        model_path = get_model_path("unitree_go2", "mjx_scene_force.xml")
        model_path = get_model_path("unitree_go2", "mjx_scene_force_crate.xml")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        for body_id in range(self.model.nbody):
            # Get the body name by its ID
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            print(f"Body {body_id}: {body_name}")
        
        # Set initial state
        self.update_state(self.current_t)

    def update_state(self, t):
        # Update MuJoCo state from stored states
        state = self.states[t]
        dims = (1, self.model.nq, self.model.nv, self.model.nu)
        dims = np.cumsum(dims)[:-1]
        i, qpos, qvel, ctrl = np.split(state, dims)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = ctrl
        mujoco.mj_forward(self.model, self.data)

    def create_trajectory_geoms(self, viewer, t):
        # Clear previous geoms
        viewer.user_scn.ngeom = 0
        cnt = 0

        white = np.array([0.8, 0.8, 0.8, 0.5])
        red = np.array([1.0, 0.0, 0.0, 1.0])
        # For each diffusion step
        for d in range(self.n_diffuse):
            # For each prediction step (except last one to connect points)
            for h in range(self.H):
                # For each body
                for b in self.vis_body_ids:
                    # Create color based on diffusion step
                    color = white * (1 - d / self.n_diffuse) + red * (d / self.n_diffuse)

                    # Get current and next positions
                    pos_current = self.predictions[t, d, h, b]
                    pos_next = self.predictions[t, d, h + 1, b]

                    # Create connector between positions
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[cnt],
                        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        size=np.array([0.0, 0.0, 0.0]),
                        pos=pos_current,
                        mat=np.eye(3).flatten(),
                        rgba=color
                    )
                    mujoco.mjv_makeConnector(
                        viewer.user_scn.geoms[cnt],
                        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        width=0.01,
                        a0=pos_current[0],
                        a1=pos_current[1],
                        a2=pos_current[2],
                        b0=pos_next[0],
                        b1=pos_next[1],
                        b2=pos_next[2],
                    )
                    viewer.user_scn.geoms[cnt].rgba = color
                    cnt += 1

        viewer.user_scn.ngeom = cnt

    def run(self):
        key_queue = queue.Queue()

        def key_callback(keycode):
            key_queue.put(keycode)


        with mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False, key_callback=key_callback
        ) as viewer:
            
            # Initial visualization
            # self.create_trajectory_geoms(viewer, self.current_t)
            # viewer.sync()
            
            while viewer.is_running():
                # Handle keyboard input
                while not key_queue.empty():
                    keycode = key_queue.get()
                    if keycode == mujoco.viewer.glfw.KEY_RIGHT:  # Right arrow
                        self.current_t = (self.current_t + 1) % self.T
                    elif keycode == mujoco.viewer.glfw.KEY_LEFT:  # Left arrow
                        self.current_t = (self.current_t - 1) % self.T

                self.current_t = (self.current_t + 1) % self.T

                # Render
                self.update_state(self.current_t)
                self.create_trajectory_geoms(viewer, self.current_t)
                viewer.sync()
                time.sleep(0.02 * 5)

if __name__ == "__main__":
    dir = "unitree_go2_crate_climb_grad"
    time_stamp = "20241130-192112"
    predictions_path = f"{dir}/{time_stamp}_predictions.npy"
    states_path = f"{dir}/{time_stamp}_states.npy"
    
    viz = PredictionViz(predictions_path, states_path)
    viz.run()