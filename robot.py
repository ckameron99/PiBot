import os

class robot():
    def __init__(self):
        self.api=api.API(self)

    def powerOptions(self,option):
        if option=='reboot':
            os.system("sudo reboot")
        if option=='powerDown':
            os.system("sudo poweroff")
        if option=='update':
            os.system("git pull && sudo reboot")

    def connectRemote(self,option):
        pass

    def connectLidar(self,option):
        pass

    def connectOrientation(self,option):
        pass

    def connectMotors(self,option):
        pass
