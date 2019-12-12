# GPU Proof of Spacemesh Time Init (aka Smeshing Setup) Prototype

## Current functionality
- A c libraray implementing the POST API setup method for cpu, cuda and openCL compute platforms.

## System Requirements

### Windows
- Windows 10 Pro.
- Microsoft Visual Studio 2017 (any edition should be okay. Visual Studio 2019 is not supported. You may also need to install specific versions of the Windows SDK when prompted when attempting to build for the first time.
- NVIDIA GPU Computing Toolkit 10.0 (but not later versions), and an NVIDIA GPU supporting CUDA 10.0 computation for CUDA testing.
- An AMD GPU supporting OpenCL 1.2 or newer for OpenCL testing.

### Linux
- Modern 64-bit Linux, such as Ubuntu, Debian.
- NVIDIA GPU Computing Toolkit 9 or 10, and an NVIDIA GPU supporting CUDA 9 or 10 computation for CUDA testing.
- An AMD GPU supporting OpenCL 1.2 or newer for OpenCL testing. Install the AMD Linux driver package (with the opencl option selected).

## Building

Build options:
SPACEMESHCL   "Build with OpenCL support" default: ON
SPACEMESHCUDA "Build with CUDA support"   default: ON

### Windows
1. Open project folder into Visual Studio 2017: `File -> Open -> Folder`.
2. Set "x64-Release" Project Settings.
3. Build: `CMake -> Rebuild All`.
4. Run test: `CMake -> Debug from Build Folder -> gpu-setup-test.exe`

### Linux
1. Create build directory:
```
  cd gpu-post
  mkdir build
  cd build
```
2. Configure:
```
  cmake ..
```
Disable OpenCL:
```
  cmake .. -DSPACEMESHCL=OFF
```
Disable CUDA:
```
  cmake .. -DSPACEMESHCUDA=OFF
```
3. Build:
```
  make
```
4. Run test:
```  
  ./test/gpu-setup-test
```

For configure with CUDA 9:
```
  cmake .. -DCMAKE_C_COMPILER=gcc-6 -DCMAKE_CXX_COMPILER=g++-6
```
You may need to set CUDA_TOOLKIT_ROOT_DIR:
```
  cmake .. -DCMAKE_C_COMPILER=gcc-6 -DCMAKE_CXX_COMPILER=g++-6 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0
```

## API

```
int scryptPositions(
    const uint8_t *id,			// 32 bytes
    uint64_t start_position,	// e.g. 0 
    uint64_t end_position,		// e.g. 49,999
    uint8_t hash_len_bits,		// (1...8) for each hash output, the number of prefix bits (not bytes) to copy into the buffer
    const uint8_t *salt,		// 32 bytes
    uint32_t options,			// throttle etc.
    uint8_t *out,				// memory buffer large enough to include hash_len_bits * number of requested hashes
    uint32_t N,					// scrypt N
    uint32_t R,					// scrypt r
    uint32_t P					// scrypt p
);
```

return to the client the system GPU capabilities. E.g. OPENCL, CUDA/NVIDIA or NONE
```
int stats();
```

stop all GPU work and don’t fill the passed-in buffer with any more results.
```
int stop(
	uint32_t ms_timeout			// timeout in milliseconds
);
```

return count of GPUs
```
int spacemesh_api_get_gpu_count(
	int type,					// GPU type SPACEMESH_API_CUDA or SPACEMESH_API_OPENCL
	int only_available			// return count of available GPUs only
);
```

lock GPU for persistent exclusive use. returned cookie used as options in scryptPositions call
```
int spacemesh_api_lock_gpu(
	int type					// GPU type SPACEMESH_API_CUDA or SPACEMESH_API_OPENCL
);
```

unlock GPU, locked by previous spacemesh_api_lock_gpu call
```
void spacemesh_api_unlock_gpu(
	int cookie					// cookie, returned by previous spacemesh_api_lock_gpu call
);
```

## Initial Benchmarks

Scrypt Benchmarks (n=512, r=1, p=1) 1 byte per leaf, batch size leaves per API call.

| Date       	| Reporter 	| impl      	| cpu / gpu                        	| Host OS             	| notes                                  	| kh/s  	| mh/s 	| x factor over 1 4ghz cpu native thread 	| x factor over 12 4ghz cpu native threads 	|
|------------	|----------	|-----------	|----------------------------------	|---------------------	|----------------------------------------	|-------	|------	|----------------------------------------	|------------------------------------------	|
| 11/19/2019 	| ae       	| go-scrypt 	| mbp + Intel i9 @ 2.9ghz - 1 core 	| OS X                	| go scrypt crypto lib (not scrypt-jane) 	| 7     	| 0.01 	| 1                                      	| 1                                        	|
| 11/19/2019 	| ae       	| sm-scrypt 	| Ryzen 5 2600x @ 4ghz - 1 core    	| Windows 10          	| scrypt-jane c code                     	| 7     	| 0.01 	| 1                                      	| 1                                        	|
| 11/19/2019 	| ae       	| sm-scrypt 	| Nvidia Gefroce RTX 2070 8GB      	| Windows 10          	| pre-optimized prototype                	| 1,920 	| 1.92 	| 290                                    	| 24.17                                    	|
| 11/19/2019 	| ae       	| sm-scrypt 	| AMD Radeon RX 580                	| Windows 10          	| pre-optimized prototype                	| 500   	| 0.50 	| 76                                     	| 6.29                                     	|
| 11/19/2019 	| ar       	| sm-scrypt 	| Nvidia GTX 1060 6G               	| Windows 10          	| pre-optimized prototype                	| 979   	| 0.98 	| 148                                    	| 12.32                                    	|
| 11/19/2019 	| ar       	| sm-scrypt 	| AMD Radeon 570 4GB               	| Windows 10          	| pre-optimized prototype                	| 355   	| 0.36 	| 54                                     	| 4.47                                     	|
| 11/12/2019 	| ae       	| sm-scrypt 	| AMD Radeon RX 580                	| Windows 10          	| optimized prototype                    	| 926   	| 0.93 	| 140                                    	| 11.65                                    	|
| 11/12/2019 	| ae       	| sm-scrypt 	| AMD Radeon RX 580                	| Ubuntu 18.0.4.3 LTS 	| optimized prototype                    	| 893   	| 0.89 	| 135                                    	| 11.24                                    	|
| 11/12/2019 	| ae       	| sm-scrypt 	| Nvidia Gefroce RTX 2070 8GB      	| Ubuntu 19.10 LTS    	| optimized prototype                    	| tbd   	| tbd  	| tbd                                    	| tbd                                      	|
