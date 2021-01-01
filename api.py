import selfupdate
import socket
import threading

class API():
    def __init__(self,robot):
        self.robotIP="192.168.4.1"
        self.appIP=None
        self.listenPort=5552
        self.sendPort=5551
        self.robot=robot
        self.connected=False
        self.listeningDaemon=threading.Thread(target=self.listen, daemon=True, args=(1,))

    def attemptConnect(self):
        if not self.connected:
            self.listeningDaemon=threading.Thread(target=self.listen, daemon=True, args=(1,))
            self.connected=True
            try:
                self.sendData(b'initConnection: True')
                self.connected=True
                return True
            except ConnectionRefusedError:
                self.connected=False
                return False
        return True

    def listen(self, name):
        s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((self.appID, self.listenPort))
        s.listen
        while True:
            data=''
            conn,addr=s.accept()
            with conn:
                while True:
                    newData=conn.recv(1024)
                    if newData:
                        data+=newData
                    else:
                        break
            self.processData(data)

    def processData(self, data):
        splitIndex=data.index(b': ')
        key=data[:splitIndex]
        params=data[splitIndex+2:]
        funcs={
        b'powerOptions': robot.powerOptions
        }
        funcs[key](params)

    def sendData(self, data):
        if self.connected:
            s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((self.appIP, self.sendPort))
            s.sendall(data)
            s.close()
            return True
        else:
            self.ui.addText(b'Robot is not connected')
            return False
