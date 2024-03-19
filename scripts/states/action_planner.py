import rospy
import smach

from geometry_msgs.msg import Pose, Point, Quaternion


class ActionTrajectory(smach.State):
    def __init__(self):
        # Your state initialization goes here
        smach.State.__init__(
            self,
            outcomes=["succeeded", "unknown_action"],
            input_keys=["action", "target_x", "target_y", "target_z"],
            output_keys=[
                "planning_group",
                "pose",
            ],  # TODO: change from one pose to list
        )

    def execute(self, userdata):
        is_right = userdata.target_y < 0
        # calculate target for action
        target_pose = Pose()
        if userdata.action == "touch":
            target_pose.position = Point(
                userdata.target_x - 0.03, userdata.target_y, userdata.target_z
            )
            if is_right:  # x, y, z, w
                target_pose.orientation = Quaternion(-0.7071068, 0, 0, 0.7071068)
            else:
                target_pose.orientation = Quaternion(0.7071068, 0, 0, 0.7071068)
        else:
            rospy.loginfo(f"Action '{userdata.action}' not defined")
            return "unknown_action"
        # set output userdata
        userdata.planning_group = "r_arm" if is_right else "l_arm"
        userdata.pose = target_pose
        return "succeeded"
