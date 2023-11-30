define laser ranger
(
    sensor
    (
        range [ 0.0 12.0 ]
        fov 360
        samples 720
    )
    # generic model properties
    color "black"
    size [ 0.06 0.15 0.03 ]
)

define diff_drive_robot position
(
    pose [ 0.0 0.0 0.0 0.0 ]
    localization "odom"
    odom_error [0.03 0.03 999999 999999 999999 0.02]

    size [0.9 0.6 0.40 ]
    origin [0.0 0.0 0.0 0.0]
    gui_nose 1

    # kinematics
    drive "diff"

    # spawn sensors
    laser(pose [0.0 0.0 -0.2 0.0 ])
)