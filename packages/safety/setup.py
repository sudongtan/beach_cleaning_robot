from setuptools import find_packages, setup

package_name = 'safety'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # thresholds live in the repo's config/ (CLAUDE.md: never in code)
        ('share/' + package_name + '/config', ['../../config/safety.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Frances Tan',
    maintainer_email='frances.tanx@gmail.com',
    description='Rule-based safety supervisor: the only publisher of /cmd_vel',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'supervisor = safety.supervisor_node:main',
        ],
    },
)
