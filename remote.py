import remote
import bluepy
import struct

class remote():
    def __init__(self,robot=None):
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

    def debugInput(self): #remove for final production. For use to reverse engineer protocol
        services=self.gamesir.getServices()
        services=list(services)
        for s in [services[2]]: # only use the second service?
            #sList=list(s)
            controlServices=s
            charac_dict=controlServices.getCharacteristics() # bluetooth characteristincs

            while True:
                charac1, charac2, charac3 = [charac.read() for charac in charac_dict]
                status_code = struct.unpack('H', charac1[:2])[0] #extract binaire and do the handling
                #print(status_code)

                if status_code == 50593:
                    on_press_key = struct.unpack('I', charac1[9:13])[0] # key value, each key on the joystick has a fixed value.

                    bar_status = struct.unpack('5B', charac1[2:7])
                    bar_status_bin = ''.join([bin(item).split('b')[1].rjust(8).replace(' ', '0') for item in bar_status])

                    left_drag = int(bar_status_bin[0:10], 2)
                    left_push = int(bar_status_bin[10:20], 2)
                    right_drag = int(bar_status_bin[20:30], 2)
                    right_push = int(bar_status_bin[30:40], 2)



                    #DEBUG and PRINT value from joystick

                    print("status %s" % status_code, end='  ')
                    print("on_press %s" % on_press_key, end='  ')
                    #print("press_counter %s" % press_counter, end='  ') #-
                    print("left_drag %s" % left_drag, end='  ')
                    print("right_drag %s" % right_drag, end='  ')
                    print("left_push %s" % left_push, end='  ')
                    print("right_push %s" % right_push, end='\r')




def main():
    r=remote()
    print(r.connect())
    r.debugInput()

if __name__=="__main__":
    main()
