from setuptools import find_packages, setup

package_name = 'localization'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Frances Tan',
    maintainer_email='frances.tanx@gmail.com',
    description='Localization implementations (baseline: ground truth)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ground_truth = localization.ground_truth:main',
        ],
    },
)
