from typing import List
import gym
import numpy as np
import ctypes
import pycuda.autoinit  # Initialize pycuda
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))  # 2D continuous action space
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))  # 4D continuous observation space
        # Load the CUDA library
        self.cuda_lib = ctypes.CDLL('./calculator.so')  # Update with the correct path
        self.setup()
        self.reset()

    def reset(self):
        initstate=np.array([1.0,0.25,0.0,0.0],dtype=np.float64)
        #self.state = (np.random.rand(4)*0.25).astype(np.float64)  # Random state between -1 and 1
        self.state=np.zeros(512*4,dtype=np.float64)
        for i in range(512):
            for j,item in enumerate(initstate):
                self.state[i*4+j]=item
        
        return self.state

    def step(self, action):
        print("Env step started")
        print(action)
        # Define the function prototype
        self.cuda_lib.call.argtypes = [ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]
        self.cuda_lib.call.restype = None
        
        # Use PyCUDA to allocate GPU memory
        state_gpu = gpuarray.to_gpu(self.state)
        cp_gpu = gpuarray.to_gpu(self.cp)
        action_gpu = gpuarray.to_gpu(action)
        
        self.cuda_lib.call(ctypes.cast(state_gpu.ptr, ctypes.POINTER(ctypes.c_double)), ctypes.cast(cp_gpu.ptr, ctypes.POINTER(ctypes.c_double)), ctypes.cast(action_gpu.ptr, ctypes.POINTER(ctypes.c_double)))
        
        self.state = gpuarray_to_numpy(state_gpu)
        #print(self.state)
        #self.state = np.array(state_gpu, dtype=np.float64)
        reward = np.zeros(512,dtype=np.float64)
        # Calculate reward based on a combination of state values (modify as needed)
        for i in range(len(reward)-1):
            reward[i]=-abs(self.state[i*4+1]-0.1)
        done = np.full(512,False)
        info = {}
        return self.state, reward, done, info

    def render(self):
        pass
    
    def setup(self):
        #  MATERIAL PROPERTIES (SI Untis)
        PV  = 0.0       # Vapour Pressure [Pa]
        RHO = 998.0     # Liquod Density [kg/m**3]
        ST  = 0.0725    # Surface Tension [N/m]
        VIS = 0.001     # Liquid viscosity [Pa s]
        CL  = 1500      # Liqid Sound Speed
        P0  = 1.0*1e5   # Ambient Pressure [Pa]
        PE  = 1.4       # Polytrophic Exponent [-]
        FREQ=31.25e3         # Hz
        R0=44.8e-6         # [m]
        
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
        
def gpuarray_to_numpy(gpu_array):
    """
    Copies data from a pycuda GPUArray to a regular NumPy array.

    Args:
        gpu_array: The GPUArray object to be converted.

    Returns:
        A NumPy array containing the data from the GPUArray.
    """

    # Get the size and data type of the GPUArray
    shape = gpu_array.shape
    dtype = gpu_array.dtype

    # Allocate memory for the NumPy array on the host (CPU)
    host_array = np.empty(shape, dtype=dtype)

    # Transfer data from GPU to host memory
    cuda.memcpy_dtoh(host_array, gpu_array.ptr)

    return host_array


def main():
    env=CustomEnv()
    action=np.array([0.225 * 100000.0,0.0]).astype(np.float64)
    env.reset()
    for i in range(4):
        env.step(action)
        
    
    
    

if __name__=="__main__":
    main()