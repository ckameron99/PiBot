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
        self.listeningDaemon.start()

    def initConnection(self,i):
        self.connected=i==b'True'
        self.sendData(b'connectionStatus: True')


    def listen(self, name):
        s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((self.robotIP, self.listenPort))
        s.listen()
        while True:
            data=b''
            conn,addr=s.accept()
            self.appIP=addr[0]
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
        b'powerOptions': robot.powerOptions,
        b'initConnection': self.initConnection
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
