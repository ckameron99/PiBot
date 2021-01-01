import api
import time
class robot():
    def __init__(self):
        self.api=api.API(self)

    def powerOptions(self,option):
        pass
r=robot()
while True:
    time.sleep(10)
