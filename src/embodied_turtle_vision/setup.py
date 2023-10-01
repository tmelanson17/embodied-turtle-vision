from setuptools import find_packages, setup

package_name = 'embodied_turtle_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='noisebridge',
    maintainer_email='noisebridge@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'computer_vision_subscriber = embodied_turtle_vision.computer_vision_subscriber:main',
        ],
    },
)
