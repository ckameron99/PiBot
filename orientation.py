import board
import busio
import adafruit_bno055


class Orientation:
    def __init__(self, robot):
        self.robot = robot
        self.started = False

    def start(self):
        if started:
            return 1
        i2c = busio.I2C(board.SCL, board.SDA)
        try:
            self.bno = adafruit_bno055.BNO055_I2C(i2c)
            self.started = True
        except:
            raise RuntimeError("BNO not connected")


if __name__ == "__main__":
    print("running")
    compass = Orientation(None)
    compass.start()