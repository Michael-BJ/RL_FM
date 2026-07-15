from setuptools import find_packages, setup

package_name = 'fm_planner'

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
            'launch/fm_planning_unknown.launch.py',
            'launch/expert_planning.launch.py',
            'launch/px4_pluginlists.yaml',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='michael',
    maintainer_email='michaelbryanjahanto@gmail.com',
    description='FM-Planner deployment package (NN/FM inference + expert data collection)',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'px4_sensor_reader             = fm_planner.px4_sensor_reader:main',
            'gz_depth_bridge_node          = fm_planner.gz_depth_bridge_node:main',
            'gz_depth_viewer_node          = fm_planner.gz_depth_viewer_node:main',
            'vfh_obstacle_avoidance_node   = fm_planner.vfh_realsense:main',
            'mavros_tf_broadcaster_node    = fm_planner.mavros_tf_broadcaster_node:main',
            'depth_to_pointcloud_node      = fm_planner.depth_to_pointcloud_node:main',
            'legacy_inference_node         = fm_planner.legacy_inference_node:main',
            'nn_inference_node             = fm_planner.nn_inference_node:main',
            'fm_inference_node             = fm_planner.fm_inference_node:main',
            'publish_pcd_node              = fm_planner.publish_pcd_node:main',
            'expert_planner_node           = fm_planner.expert_planner_node:main',
        ],
    },
)
