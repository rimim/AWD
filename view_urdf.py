import argparse
import placo
import time
import threading
import webbrowser
from placo_utils.visualization import *

def open_browser():
    try:
        webbrowser.open_new('http://127.0.0.1:7000/static/')
    except:
        print("Failed to open the default browser. Trying Google Chrome.")
        try:
            webbrowser.get('google-chrome').open_new('http://127.0.0.1:7000/static/')
        except:
            print("Failed to open Google Chrome. Make sure it's installed and accessible.")

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("path", help="Path to the URDF")
arg_parser.add_argument("--frames", help="Frame to display", nargs="+")
arg_parser.add_argument("--animate", help="Animate the robot", action="store_true")
args = arg_parser.parse_args()

start_time = time.perf_counter()
robot = placo.RobotWrapper(args.path, placo.Flags.ignore_collisions)
robot.update_kinematics()
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"loading {args.path} took {elapsed_time:.6f} seconds.")

print("Joint names:")
print(list(robot.joint_names()))

print("Frame names:")
print(list(robot.frame_names()))

viz = robot_viz(robot)
t = 0
threading.Timer(1, open_browser).start()

while True:
    viz.display(robot.state.q)

    if args.frames:
        for frame in args.frames:
            robot_frame_viz(robot, frame)

    if args.animate:
        for joint in robot.joint_names():
            robot.set_joint(joint, np.sin(t))
            robot.update_kinematics()

    t += 0.01
    time.sleep(0.01)