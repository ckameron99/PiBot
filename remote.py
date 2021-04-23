import bluepy
import struct
import _thread as th



class DemoStuff:
    def __init__(self):

        self.r = remote()
        self.r.connect()
        th.start_new_thread(self.r.ConstantUpdate, ())

        """#### DO Stufff
        if self.r.StuffBeingPressed["A"]:
            pass
            #Do stuff
        ### More stuff
        
        # Example with Function calling
        self.x = 0
        def add():
            self.x += 1
        
        r.FuncsToDo["A"] = add"""
        
        



class remote:
    def __init__(self, robot=None):

        self.connected = False
        self.robot = robot

        self.StuffBeingPressed = {"A": False, "B": False, "Y": False, "X": False, "L1": False, "L2": False,
                                  "R1": False, "R2": False, "C1": False, "C2": False, "Menu": False, "UP-D": False,
                                  "L-Stick": (0, 0), "R-Stick": (0, 0)}
        
        self.FuncsToDo = {"A": None, "B": None, "Y": None, "X": None, "L1": None, "L2": None,
                                  "R1": None, "R2": None, "C1": None, "C2": None, "Menu": None, "UP-D": None,
                                  "L-Stick": None, "R-Stick": None}

    def connect(self):
        if self.connected:
            return True
        try:
            self.gamesir = bluepy.btle.Peripheral("C6:86:A1:02:16:82", 'random')
            self.connected = True
            return True
        except bluepy.btle.BTLEDisconnectError:
            return False

    def ConstantUpdate(self):
        services = self.gamesir.getServices()
        services = list(services)
        for s in [services[2]]:  # only use the second service?
            # sList=list(s)
            controlServices = s
            charac_dict = controlServices.getCharacteristics()  # bluetooth characteristincs

            while True:
                charac1, charac2, charac3 = [charac.read() for charac in charac_dict]
                status_code = struct.unpack('H', charac1[:2])[0]  # extract binaire and do the handling
                # print(status_code)

                if status_code == 50593:
                    on_press_key = struct.unpack('I', charac1[9:13])[
                        0]  # key value, each key on the joystick has a fixed value.

                    bar_status = struct.unpack('5B', charac1[2:7])
                    bar_status_bin = ''.join(
                        [bin(item).split('b')[1].rjust(8).replace(' ', '0') for item in bar_status])

                    left_drag = int(bar_status_bin[0:10], 2)
                    left_push = int(bar_status_bin[10:20], 2)
                    right_drag = int(bar_status_bin[20:30], 2)
                    right_push = int(bar_status_bin[30:40], 2)

                    if (on_press_key & 1) != 0:
                        self.StuffBeingPressed["A"] = True
                        if self.FuncsToDo["A"] is not None:
                            self.FuncsToDo["A"]()
                        # print("A pressed")
                    else:
                        self.StuffBeingPressed["A"] = False

                    if (on_press_key & 2) != 0:
                        self.StuffBeingPressed["B"] = True
                        if self.FuncsToDo["B"] is not None:
                            self.FuncsToDo["B"]()
                        # print("B pressed")
                    else:
                        self.StuffBeingPressed["B"] = False

                    if (on_press_key & 16) != 0:
                        self.StuffBeingPressed["Y"] = True
                        if self.FuncsToDo["Y"] is not None:
                            self.FuncsToDo["Y"]()
                        # print("Y pressed")
                    else:
                        self.StuffBeingPressed["Y"] = False

                    if (on_press_key & 8) != 0:
                        self.StuffBeingPressed["X"] = True
                        if self.FuncsToDo["X"] is not None:
                            self.FuncsToDo["X"]()
                        # print("X pressed")
                    else:
                        self.StuffBeingPressed["X"] = False

                    if (on_press_key & 64) != 0:
                        self.StuffBeingPressed["L1"] = True
                        # print("L1 pressed")
                    else:
                        self.StuffBeingPressed["L1"] = False

                    if (on_press_key & 256) != 0:
                        self.StuffBeingPressed["L2"] = True
                        # print("L2 pressed")
                    else:
                        self.StuffBeingPressed["l2"] = False

                    if (on_press_key & 128) != 0:
                        self.StuffBeingPressed["R1"] = True
                        # print("R1 pressed")
                    else:
                        self.StuffBeingPressed["R1"] = False

                    if (on_press_key & 512) != 0:
                        self.StuffBeingPressed["R2"] = True
                        # print("R2 pressed")
                    else:
                        self.StuffBeingPressed["R2"] = False

                    if (on_press_key & 1024) != 0:
                        self.StuffBeingPressed["C1"] = True
                        # print("C1 pressed")
                    else:
                        self.StuffBeingPressed["C1"] = False

                    if (on_press_key & 2048) != 0:
                        self.StuffBeingPressed["C2"] = True
                        # print("C2 pressed")
                    else:
                        self.StuffBeingPressed["C2"] = False
                    if (on_press_key & 4) != 0:
                        self.StuffBeingPressed["Menu"] = True
                        # print("Menu pressed")
                    else:
                        self.StuffBeingPressed["Menu"] = False

                    if (on_press_key & 65536) != 0:
                        self.StuffBeingPressed["UP-D"] = True
                        # print("Up-DPAD pressed")
                    else:
                        self.StuffBeingPressed["UP-D"] = False

                    if left_push != 512 or left_drag != 512:
                        x = (left_drag - 512) / 512
                        y = (512 - left_push) / 512
                        self.StuffBeingPressed["L-Stick"] = (x,y)
                        #print(f'Vector direction of lstick is ({x},{y})')
                    else:
                        self.StuffBeingPressed["L-Stick"] = (0,0)

                    if right_push != 512 or right_drag != 512:
                        x = (right_drag - 512) / 512
                        y = (512 - right_push) / 512
                        self.StuffBeingPressed["R-Stick"] = (x,y)
                        #print(f'Vector direction of lstick is ({x},{y})')
                    else:
                        self.StuffBeingPressed["R-Stick"] = (0,0)
                    # DEBUG and PRINT value from joystick

    def UsefulldebugInput(self):  # remove for final production. For use to reverse engineer protocol
        services = self.gamesir.getServices()
        services = list(services)
        for s in [services[2]]:  # only use the second service?
            # sList=list(s)
            controlServices = s
            charac_dict = controlServices.getCharacteristics()  # bluetooth characteristincs

            while True:
                charac1, charac2, charac3 = [charac.read() for charac in charac_dict]
                status_code = struct.unpack('H', charac1[:2])[0]  # extract binaire and do the handling
                # print(status_code)

                if status_code == 50593:
                    on_press_key = struct.unpack('I', charac1[9:13])[
                        0]  # key value, each key on the joystick has a fixed value.

                    bar_status = struct.unpack('5B', charac1[2:7])
                    bar_status_bin = ''.join(
                        [bin(item).split('b')[1].rjust(8).replace(' ', '0') for item in bar_status])

                    left_drag = int(bar_status_bin[0:10], 2)
                    left_push = int(bar_status_bin[10:20], 2)
                    right_drag = int(bar_status_bin[20:30], 2)
                    right_push = int(bar_status_bin[30:40], 2)

                    if (on_press_key & 1) != 0:
                        print("A pressed")

                    if (on_press_key & 2) != 0:
                        print("B pressed")

                    if (on_press_key & 16) != 0:
                        print("Y pressed")

                    if (on_press_key & 8) != 0:
                        print("X pressed")

                    if (on_press_key & 64) != 0:
                        print("L1 pressed")

                    if (on_press_key & 256) != 0:
                        print("L2 pressed")

                    if (on_press_key & 128) != 0:
                        print("R1 pressed")

                    if (on_press_key & 512) != 0:
                        print("R2 pressed")

                    if (on_press_key & 1024) != 0:
                        print("C1 pressed")

                    if (on_press_key & 2048) != 0:
                        print("C2 pressed")

                    if (on_press_key & 4) != 0:
                        print("Menu pressed")

                    if (on_press_key & 65536) != 0:
                        print("Up-DPAD pressed")

                    if left_push != 512 or left_drag != 512:
                        x = (left_drag - 512) / 512
                        y = (512 - left_push) / 512
                        self.StuffBeingPressed["L-Stick"] = (x, y)
                        # print(f'Vector direction of lstick is ({x},{y})')
                    else:
                        self.StuffBeingPressed["L-Stick"] = (0, 0)

                    if right_push != 512 or right_drag != 512:
                        x = (right_drag - 512) / 512
                        y = (512 - right_push) / 512
                        self.StuffBeingPressed["R-Stick"] = (x, y)
                        # print(f'Vector direction of rstick is ({x},{y})')
                    else:
                        self.StuffBeingPressed["R-Stick"] = (0, 0)

                    # DEBUG and PRINT value from joystick

                    '''print("status %s" % status_code, end='  ')
                    print("on_press %s" % on_press_key, end='  ')
                    #print("press_counter %s" % press_counter, end='  ') #-
                    print("left_drag %s" % left_drag, end='  ')
                    print("right_drag %s" % right_drag, end='  ')
                    print("left_push %s" % left_push, end='  ')
                    print("right_push %s" % right_push, end='\r')'''

    def debugInput(self):  # remove for final production. For use to reverse engineer protocol
        services = self.gamesir.getServices()
        services = list(services)
        for s in [services[2]]:  # only use the second service?
            # sList=list(s)
            controlServices = s
            charac_dict = controlServices.getCharacteristics()  # bluetooth characteristincs

            while True:
                charac1, charac2, charac3 = [charac.read() for charac in charac_dict]
                status_code = struct.unpack('H', charac1[:2])[0]  # extract binaire and do the handling
                # print(status_code)

                if status_code == 50593:
                    on_press_key = struct.unpack('I', charac1[9:13])[
                        0]  # key value, each key on the joystick has a fixed value.

                    bar_status = struct.unpack('5B', charac1[2:7])
                    bar_status_bin = ''.join(
                        [bin(item).split('b')[1].rjust(8).replace(' ', '0') for item in bar_status])

                    left_drag = int(bar_status_bin[0:10], 2)
                    left_push = int(bar_status_bin[10:20], 2)
                    right_drag = int(bar_status_bin[20:30], 2)
                    right_push = int(bar_status_bin[30:40], 2)

                    # DEBUG and PRINT value from joystick

                    print("status %s" % status_code, end='  ')
                    print("on_press %s" % on_press_key, end='  ')
                    # print("press_counter %s" % press_counter, end='  ') #-
                    print("left_drag %s" % left_drag, end='  ')
                    print("right_drag %s" % right_drag, end='  ')
                    print("left_push %s" % left_push, end='  ')
                    print("right_push %s" % right_push, end='\r')
