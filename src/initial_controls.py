# code from: https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/notebooks/basic_motion/basic_motion.ipynb
from jetbot import Robot

robot = Robot()


def stop():
    robot.stop()
    
def step_forward():
    robot.forward(0.4)
    time.sleep(0.5)
    robot.stop()

def step_backward():
    robot.backward(0.4)
    time.sleep(0.5)
    robot.stop()

def step_left():
    robot.left(0.3)
    time.sleep(0.5)
    robot.stop()

def step_right():
    robot.right(0.3)
    time.sleep(0.5)
    robot.stop()

# basic test to get robot moving
def test_robot():
	step_forward()
	step_backward()
	step_left()
	step_right()


def main():
    test_robot()


if __name__ == "__main__":
    main()

