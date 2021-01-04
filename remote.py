import remote
import bluepy
import struct

class remote():
    def __init__(self,robot):
        self.connected=False
        self.robot=robot
    def connect(self):
        if self.connected:
            return True
        try:
            self.gamesir=bluepy.btle.Peripheral("C6:86:A1:02:16:82", 'random')
            self.connected=True
            return True
        except bluepy.btle.BTLEDisconnectError:
            return False



def main():
    r=remote()

if __name__=="__main__":
    main()
