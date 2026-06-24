from setuptools import find_packages, setup

package_name = 'neo_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/sitl.launch.py',
            'launch/mavros_only.launch.py',
            'launch/camera_vfh.launch.py',
            'launch/nn_planning_unknown.launch.py',
            'launch/nn_inference.launch.py',
            'launch/expert_planning.launch.py',
            'launch/px4_pluginlists.yaml',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='michael',
    maintainer_email='michaelbryanjahanto@gmail.com',
    description='NEO-Planner deployment package (NN inference + expert data collection)',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'px4_sensor_reader             = neo_planner.px4_sensor_reader:main',
            'gz_depth_bridge_node          = neo_planner.gz_depth_bridge_node:main',
            'gz_depth_viewer_node          = neo_planner.gz_depth_viewer_node:main',
            'vfh_obstacle_avoidance_node   = neo_planner.vfh_realsense:main',
            'mavros_tf_broadcaster_node    = neo_planner.mavros_tf_broadcaster_node:main',
            'depth_to_pointcloud_node      = neo_planner.depth_to_pointcloud_node:main',
            'neo_planner_node              = neo_planner.neo_planner_node:main',
            'nn_inference_node             = neo_planner.nn_inference_node:main',
            'publish_pcd_node              = neo_planner.publish_pcd_node:main',
            'expert_planner_node           = neo_planner.expert_planner_node:main',
        ],
    },
)
