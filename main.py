import api
import time
class robot():
    def __init__(self):
        self.api=api.API(self)

    def powerOptions(self,option):
        pass

    def connectRemote(self,option):
        pass

    def connectLidar(self,option):
        pass

    def connectOrientation(self,option):
        pass

    def connectMotors(self,option):
        pass

        
r=robot()
while True:
    time.sleep(10)
