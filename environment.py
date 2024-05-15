from typing import List
import gym
import numpy as np
import ctypes
import pycuda.autoinit  # Initialize pycuda
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import torch

class CustomEnv(gym.vector.VectorEnv):
    def __init__(self,
                 excitation_frequecy: float = 25.0e3,
                 rest_radius: float = 80.0e-6,
                 total_steps_per_episode: int = 25,
                 timet_step_length: float = 10.0,
                 number_of_parallel_envs: int = 512,
                 actual_target_position: float = 0.1,
                 obs_dim: int = 4,
                 act_dim: int = 2):
        #self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))  # 2D continuous action space ez így nem jó, PA nem lehet -1
        action_space = gym.spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0*1e5, 0.5 * np.pi]), shape=(2, ))
        observation_space = gym.spaces.Box(low=np.array([0.0, -0.75, -500.0, -1.0]), high=np.array([10.0, 1.0, 500.0, 1.0]), shape=(4, ) )   #!!!!!!
        #self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))  # 4D continuous observation space ez így nem jó, miért lenne minden -1 és 1 között by default..
        # Load the CUDA library

        super(CustomEnv, self).__init__(number_of_parallel_envs, observation_space, action_space)
        self.cuda_lib = ctypes.CDLL('./calculator.so')  # Update with the correct path
        self.cuda_lib.call.argtypes = [ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]
        self.cuda_lib.call.restype = ctypes.c_void_p


        self.setup(excitation_frequecy, rest_radius)
        #self.reset()       usually not called here

        # Environment's static features (to avoid hardcoded numbers in member functions)
        self.total_steps_per_episode = total_steps_per_episode
        self.timet_step_length       = timet_step_length            # Nem használjuk, de mivel hardcodeolva van a CUDA kóon belül ki kell nézni
        self.number_of_parallel_envs = number_of_parallel_envs      # Nthreads-el összhangban kell lenni (hardcode a reset stb. loopokban elkerülhetü.)
        self.actual_target_position  = actual_target_position       # Egyezrübb itt definiálni, mint a rewardon belül keresgélni
        self.obs_dim=obs_dim
        self.act_dim=act_dim
        self.episodic_return=torch.zeros((number_of_parallel_envs,), dtype=np.float64)
        self.episodic_length=torch.zeros((number_of_parallel_envs,), dtype=np.int64)

    def step_cuda(self, state_gpu, action_gpu):
        self.cuda_lib.call(ctypes.cast(state_gpu.ptr, ctypes.POINTER(ctypes.c_double)), ctypes.cast(self.cp_gpu.ptr, ctypes.POINTER(ctypes.c_double)), ctypes.cast(action_gpu.ptr, ctypes.POINTER(ctypes.c_double)))

    def reset(self):
        self.initstate=np.array([1.0,0.20,0.0,0.0],dtype=np.float64)
        
        self.steps_done = torch.zeros((self.number_of_parallel_envs, ), dtype=np.int64)
        self.rewards_collected = np.zeros((self.number_of_parallel_envs, ), dtype=np.float64)

        self.state=np.tile(self.initstate, self.number_of_parallel_envs)
        return self.state

    def reset_env(self, env_id):
        self.state[4*env_id:4*env_id+3]=self.initstate
        self.steps_done[env_id]=0
        self.episodic_length[env_id]=0
        self.episodic_return=0.0
    
    def get_state(self, env_id: int):
        if env_id < self.number_of_parallel_envs:
            return np.array([self.state[env_id*4 + i] for i in range(4)])
        else:
            return None


    def step(self, action: np.ndarray):
        # Clip and flatten action
        action = np.clip(action, self.action_space.low, self.action_space.high).flatten()
        
        
        # Use PyCUDA to allocate GPU memory
        state_gpu = gpuarray.to_gpu(self.state)
        action_gpu = gpuarray.to_gpu(action)

        
        self.step_cuda(state_gpu, action_gpu)
        self.steps_done +=1
        self.episodic_length+=1
        
        
        
        self.state = gpuarray_to_numpy(state_gpu) 
        t_state=torch.reshape(torch.tensor(self.state)(self.obs_dim,self.number_of_parallel_envs))

        # teminated timeouted and their combined -should be [tensor: bool, (num_paralell,)]
        self.terminated = torch.abs(t_state[1] - self.actual_target_position) < 0.01
        self.time_out = self.steps_done == self.total_steps_per_episode
        self.ended=torch.logical_or(self.terminated, self.time_out)
        
        
        reward = self.get_reward()
        self.episodic_return+=reward
        
        
        # collect the indecies of terminated and timeouted trajectories and reset them
        dones=torch.nonzero(self.ended).squeeze()
        self.reset_env(dones)
        
        # create info log
        info = {"final_observation": t_state[:, self.ended], # collects terminated and timeouted states [torch.tensor: ( obs_dim , num_of_terminated + num_of_timeouted )]
                "dones": dones, # collected indecies
                "episodic_return": self.episodic_return[self.ended],
                "episode_length": self.episodic_length[self.ended]}
        return t_state, reward, self.terminated, self.time_out, info


    def get_reward(self, t_state):
        return torch.where(self.terminated,-torch.abs(t_state[1]-0.1)+10,-torch.abs(t_state[1]-0.1))

    def render(self):
        pass

    def setup(self, excitation_frequecy, rest_radius):
        #  MATERIAL PROPERTIES (SI Untis)
        PV  = 0.0       # Vapour Pressure [Pa]
        RHO = 998.0     # Liquod Density [kg/m**3]
        ST  = 0.0725    # Surface Tension [N/m]
        VIS = 0.001     # Liquid viscosity [Pa s]
        CL  = 1500      # Liqid Sound Speed
        P0  = 1.0*1e5   # Ambient Pressure [Pa]
        PE  = 1.4       # Polytrophic Exponent [-]

        FREQ=excitation_frequecy    # Hz
        R0=rest_radius              # [m]

        wr = 2.0 * np.pi * FREQ
        lr = CL / FREQ

        self.cp = np.zeros((19, ), dtype=np.float64)
        self.cp[0] = (2.0 * ST / R0 + P0 - PV) * pow((2.0 * np.pi / R0 / wr), 2.0) / RHO
        self.cp[1] = (1.0 - 3.0 * PE) * (2.0 * ST / R0 + P0 - PV) * (2.0 * np.pi / R0 / wr) / CL / RHO
        self.cp[2] = (P0 - PV) * pow((2.0 * np.pi / R0 / wr), 2.0) / RHO
        self.cp[3] = (2.0 * ST / R0 / RHO) * pow((2.0 * np.pi / R0 / wr), 2.0)
        self.cp[4] = 4.0 * VIS / RHO / (pow(R0, 2.0)) * (2.0 * np.pi / wr)
        self.cp[5] = (pow((2.0 * np.pi / R0 / wr), 2.0)) / RHO
        self.cp[6] = pow((2.0 * np.pi / wr), 2.0) / CL / RHO / R0
        self.cp[7] = R0 * wr / (2.0 * np.pi) / CL

        # Translational - Motion Constansts
        self.cp[8] = pow((0.5 * lr / R0), 2.0)
        self.cp[9] = (2.0 * np.pi) / RHO / R0 / lr / pow((wr * R0), 2.0)
        self.cp[10] = 4.0 * np.pi / 3.0 * pow(R0, 3.0)
        self.cp[11] = 12.0 * np.pi * VIS * R0

        # Static Constants
        self.cp[12] = 3.0 * PE
        self.cp[13] = 1.0 / wr
        self.cp[14] = lr / (2.0 * np.pi)
        self.cp[15] = CL
        self.cp[16] = 1.0 / RHO / CL

        # Dinamic Constants
        self.cp[17] = wr
        self.cp[18] = 2.0 * np.pi / lr

        self.cp_gpu = gpuarray.to_gpu(self.cp)

def gpuarray_to_numpy(gpu_array):
    
    # Get the size and data type of the GPUArray
    shape = gpu_array.shape
    dtype = gpu_array.dtype

    # Allocate memory for the NumPy array on the host (CPU)
    host_array = np.empty(shape, dtype=dtype)

    # Transfer data from GPU to host memory
    cuda.memcpy_dtoh(host_array, gpu_array.ptr)

    return host_array
