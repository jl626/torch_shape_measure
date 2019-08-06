from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
install_requires = [
    'numpy',
    'scipy',
    'networkx',
    'scikit-learn',
    'plyfile',
    'pandas',
    'rdflib',
    'h5py',
	'googledrivedownloader',
	'torch_geometric'
]

setup(
	name='PyTorch-metric',
	version='0.2',
	author='Juheon Lee',
	author_email='juheon@gmail.com', 
	description='A PyTorch module for the Wasserstein, Chamfer, current and varifold metrics', 
	install_requires=install_requires,
	ext_package='_shape_ext',
	ext_modules=[
		CUDAExtension(
			name='_metric', 
			sources=[
				'package/src/metrics.cpp',
				'package/src/emd.cu',
				'package/src/chamfer.cu',
				'package/src/varifold_kernel.cu',
			],
			include_dirs=['package/include'],
			libraries=["cusolver", "cublas"],
		),
	],
	packages=[
		'shape_metric', 
	],
	package_dir={
		'shape_metric' : 'package/layer'
	},
	cmdclass={'build_ext': BuildExtension}, 
)
