from setuptools import setup, find_packages

setup(
    name='zqdl',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'matplotlib',
        'loguru',
        'objprint',
        'Pillow',
    ],
    author='wick',
    author_email='zhou_qiang98@163.com',
    description='A package for deep learning and data science',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ZhuoQiang1998/DL',  # 你的项目主页
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
