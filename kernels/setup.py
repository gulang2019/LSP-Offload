from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os 

# device = os.environ.get('DEVICE', 'cuda')
# src = 'fusedAdam_cpu.cpp'
# src = 'fusedAdam_cuda.cu' if device == 'cuda' else 'fusedAdam_cpu.cpp'
# print(f'Using {device} device, compiling {src}')

setup(
    name='fused_adam',
    ext_modules=[
        CUDAExtension('fused_adam', [
            'fusedAdam.cpp',
            'fusedAdam_cuda.cu',
            'fusedAdam_cpu.cpp'
        ],
        extra_compile_args={'cxx': ['-fopenmp', '-mavx2'],
                            'nvcc': ['-O2']}
        # extra_compile_args={'cxx': ['-g'],
        #         'nvcc': ['-g', '-G']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
