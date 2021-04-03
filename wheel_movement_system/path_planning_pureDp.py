# this module should only be run once at the beginning
# calculation result will be stored
# very slow (21 hours) on first running, but reliable
import numpy as np
import os
import json
import queue
import math
from tqdm import tqdm


class position:
    def __init__(self, *arg):
        if len(arg) == 3:
            x, y, a = arg
            assert(isinstance(x, int) and isinstance(y, int) and isinstance(a, int))
            self.x = x
            self.y = y
            self.a = a
        elif len(arg) == 1:
            string = arg[0]
            assert(isinstance(string, str) and len(string) == 12)
            self.x = int(string[:4])
            self.y = int(string[4:8])
            self.a = int(string[8:])
            assert(isinstance(self.x, int) and isinstance(self.y, int) and isinstance(self.a, int))
    
    def __str__(self):
        s = '%04d' % self.x
        s += '%04d' % self.y
        s += '%04d' % self.a
        return s

    def val(self):
        return self.x, self.y, self.a
    
    def location(self):
        return self.x, self.y

    def __sub__(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


def build_dp_result_matrix(V_max, A_max, l, ldiv=400, adiv=360, dt=0.1, L=400, VL0=0, VR0=0):
    '''
    Args:
        V_max: float - max velocity a wheel can reach (in mm/s);
        A_max: float - max acceration a wheel can reach (in mm/s^2);
        l: float - the abeam distance between wheel and the centre (in mm);
        ldiv: unsigned int - discribes the accuracy of dividing length (larger = more precise);
        adiv: unsigned int - discribes the accuracy of dividing angle / bearing (larger = more precise);
        dt: float - the minimum unit of time (in s);
        L: float - size of the field (in mm);
        VL0, VR0: float - the speed of left-side and right-side wheels at the beginning
    Notice:
        time complexity: O(ldiv**2 * adiv * L**2 * (V_max*dt/l)**3)
        space complexity: O(ldiv**2 * adiv)
    Returns:
        1. numpy.ndarray of 3 diamentions, storing int
        If the coordinate means (x, y, bearing),
        the return t[x][y][bearing] discribes the minimum time unit needed to go from (0, 0, 0) to (x, y, bearing).
        2. dict - the last state of this state
    Boundness:
        The max acceration of the wheels is not considered, to compromise space.
    '''
    # build a sub-directory for storage (if it havn't been built before)
    sub_dir = 'cache2'
    try:
        os.makedirs(sub_dir)
    except FileExistsError:
        pass
    dp_result_file = os.path.join(sub_dir, 'dp_result.json')
    dp_argument_file = os.path.join(sub_dir, 'dp_args.json')
    dp_traceback_file = os.path.join(sub_dir, 'dp_trba.json')

    unavaliable = int(3600 / dt)
    current_argument_list = {'V_max': V_max, 'l': l, 'ldiv': ldiv, 'adiv': adiv, 'dt': dt, 'L': L, 'VL0': VL0, 'VR0': VR0, 'unavaliable': unavaliable}

    # if result with the same arguments already exists, no more calculation is needed
    try:
        with open(dp_argument_file, 'r') as arg:
            argument_list = dict(json.load(arg))
        assert(argument_list == current_argument_list)
        with open(dp_result_file, 'r') as res:
            t = np.array(json.load(res))
        with open(dp_traceback_file, 'r') as tb:
            last = dict(json.load(tb))
        return t, last
    except FileNotFoundError:
        print('This is the first-time calculation.')
    except AssertionError:
        print('Argument changed.')

    t = np.full((ldiv + 1, ldiv + 1, adiv), unavaliable)  # initialize time to every state to 1 hour, which is longer than any possible result
    q = queue.Queue()  # a state in t with value changed will get into this queue and be used later to update other states
    last = {}  # map<tuple, tuple> logs the last state of this state
    # in other words, t will be updated in topological order

    unit_l = L / ldiv
    unit_a = 2 * np.pi / adiv
    try:
        assert(A_max * dt ** 2 / 2 > 5 * unit_l)
    except AssertionError:
        raise ValueError('Wax dt, ldiv or adiv, please.')

    t[0][0][0] = 0  # this is the starting point
    q.put((0, 0, 0))

    pbar = tqdm(total=(ldiv + 1) ** 2 * adiv)  # show progressing bar
    while not q.empty():
        pbar.update(1)  # update progressing bar

        x, y, a = q.get()  # 0 <= x, y <= ldiv; 0 <= a < adiv.

        try:
            this_state = position(x, y, a)
            last_state = position(last[str(position(x, y, a))])
            Ds = (this_state - last_state) * unit_l
            Da = (this_state.a - last_state.a) * unit_a
            VL = Ds + l * Da
            VR = Ds - l * Da
        except:
            VL = VL0
            VR = VR0
        
        for SL in np.arange(max(VL ** 2 / A_max / 2, VL * dt - A_max * dt ** 2 / 2), 
                            VL * dt + A_max * dt ** 2 / 2 - max(0, VL + A_max * dt - V_max) ** 2 / A_max / 2,
                            unit_l / 2):
            for SR in np.arange(max(VR ** 2 / A_max / 2, VR * dt - A_max * dt ** 2 / 2), 
                                VR * dt + A_max * dt ** 2 / 2 - max(0, VR + A_max * dt - V_max) ** 2 / A_max / 2,
                                unit_l / 2):
                ds = (SL + SR) / 2
                da = int(round((SL - SR) / (2 * l * unit_a)))
                dx = int(round(ds * math.cos(a * unit_a) / unit_l))
                dy = int(round(ds * math.sin(a * unit_a) / unit_l))
                if 0 <= x + dx and x + dx <= ldiv and 0 <= y + dy and y + dy <= ldiv:
                    if t[x + dx][y + dy][(a + da) % adiv] > t[x][y][a] + 1:
                        t[x + dx][y + dy][(a + da) % adiv] = t[x][y][a] + 1
                        last[str(position(x + dx, y + dy, (a + da) % adiv))] = str(position(x, y, a))
                        q.put((x + dx, y + dy, (a + da) % adiv))

    pbar.close()

    # save json to file
    with open(dp_result_file, 'w') as ofile:
        json.dump(t.tolist(), ofile)
    with open(dp_argument_file, 'w') as oarg:
        json.dump(current_argument_list, oarg)
    with open(dp_traceback_file, 'w') as otb:
        json.dump(last, otb)

    return t, last


def find_path(x: int, y: int, a: int):
    '''
    In company with 'build_dp_result_matrix'
    return a list of positions forming the fastest path from (0, 0, 0) to (x, y, a)
    '''
    sub_dir = 'cache2'
    dp_result_file = os.path.join(sub_dir, 'dp_result.json')
    dp_argument_file = os.path.join(sub_dir, 'dp_args.json')
    dp_traceback_file = os.path.join(sub_dir, 'dp_trba.json')

    try:
        with open(dp_argument_file, 'r') as arg:
            argument_list = dict(json.load(arg))
        with open(dp_result_file, 'r') as res:
            t = np.array(json.load(res))
        with open(dp_traceback_file, 'r') as tb:
            last = dict(json.load(tb))

        assert(isinstance(x, int))
        assert(isinstance(y, int))
        assert(isinstance(a, int))
        assert(x >= 0)
        assert(y >= 0)
        assert(a >= 0)
        assert(x <= int(argument_list['ldiv']))
        assert(y <= int(argument_list['ldiv']))
        assert(a < int(argument_list['adiv']))

        if t[x][y][a] == argument_list['unavaliable']:
            raise ValueError
    except FileNotFoundError:
        raise RuntimeError('Should run \'build_dp_result_matrix\' first.')
    except AssertionError:
        print('State not calculated.')
        return None
    except ValueError:
        print('Can\'t reach this state.')
        return None

    stack = queue.LifoQueue()
    current_state = position(x, y, a)
    while str(current_state) != '000000000000':
        stack.put(current_state)
        current_state = position(last[str(current_state)])
    stack.put(position(0, 0, 0))
    
    path = []
    while not stack.empty():
        path.append(stack.get())

    return path
    