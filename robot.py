import os
import remote
import api
import time

class robot():
    def __init__(self):
        self.api=api.API(self)
        self.remote=remote.remote()

    def powerOptions(self,option):
        if option=='reboot':
            os.system("sudo reboot")
        if option=='powerDown':
            os.system("sudo poweroff")
        if option=='update':
            os.system("git pull && sudo reboot")

    def connectRemote(self,option):
        self.remote.connect()
        if self.remote.connected:
            self.api.sendData(b'remoteStatus: SUCCESS')
        else:
            self.api.sendData(b'remoteStatus: FAILED')

    def connectLidar(self,option):
        pass

    def connectOrientation(self,option):
        pass

    def connectMotors(self,option):
        pass
