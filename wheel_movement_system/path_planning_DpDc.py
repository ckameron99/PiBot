# this module should only be run once at the beginning
# calculation result will be stored
# dynamic programming and divide-and-conquer are used
# faster than pure dp, but not tested yet
import numpy as np
import os
import json
import queue
import math
from tqdm import tqdm


class position:
    def __init__(self, *arg):
        if len(arg) == 3:
            r, p, a = arg
            assert(isinstance(r, int) and isinstance(p, int) and isinstance(a, int))
            self.r = r
            self.p = p
            self.a = a
        elif len(arg) == 1:
            string = arg[0]
            assert(isinstance(string, str))

            if len(string) == 12:
                self.r = int(string[:4])
                self.p = int(string[4:8])
                self.a = int(string[8:])
            elif len(string) == 9:
                self.r = int(string[:3])
                self.p = int(string[3:6])
                self.a = int(string[6:])
            elif len(string) == 8:
                self.r = int(string[:2])
                self.p = int(string[2:5])
                self.a = int(string[5:])
            elif len(string) == 6:
                self.r = int(string[:2])
                self.p = int(string[2:4])
                self.a = int(string[4:])
            else:
                raise ValueError('unsupported encoding system')
        
            assert(isinstance(self.r, int) and isinstance(self.p, int) and isinstance(self.a, int))
    
    def __str__(self):
        s = '%02d' % self.r
        s += '%02d' % self.p
        s += '%02d' % self.a

        if len(s) > 6:
            s = '%02d' % self.r
            s += '%03d' % self.p
            s += '%03d' % self.a

        if len(s) > 8:
            s = '%03d' % self.r
            s += '%03d' % self.p
            s += '%03d' % self.a

        if len(s) > 9:
            s = '%04d' % self.r
            s += '%04d' % self.p
            s += '%04d' % self.a
        
        if len(s) > 12:
            raise ValueError('unexpected large value')

        return s

    def val(self):
        return self.r, self.p, self.a
    
    def location_p(self):
        return self.r, self.p

    def location_c(self):
        '''Still need to time unit_l before use'''
        real_p = 2 * np.pi / int(round(2 * np.pi * self.r)) * self.p
        x = self.r * np.cos(real_p)
        y = self.r * np.sin(real_p)
        return x, y

    def __sub__(self, other):
        x0, y0 = self.location_c()
        x1, y1 = other.location_c()
        return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

    def is_zero(self):
        if self.r == 0 and self.p == 0 and self.a == 0:
            return True
        else:
            return False


def build_level_0(sub_dir, V_max, l, ldiv, adiv, accuracy, dt, unavaliable):
    dp_result_file = os.path.join(sub_dir, 't%d.json' % 0)
    dp_traceback_file = os.path.join(sub_dir, 'trba%d.json' % 0)

    # initialize time to every state to 1 hour, which is longer than any possible result
    # t0[a][r][p] (theta, r, phi)
    t0 = np.full((adiv, ldiv + 1), None)
    for t0_a in t0:
        t0_a[0] = np.array([unavaliable])
        for k in range(1, ldiv + 1):
            t0_a[k] = np.full(int(round(2 * np.pi * k)), unavaliable)
    
    # now t0[r][p][a] as before
    t0 = t0.transpose(1, 2, 0)

    q = queue.Queue()  # a state in t with value changed will get into this queue and be used later to update other states
    last = {}  # map<tuple, tuple> logs the last state of this state
    # in other words, t will be updated in topological order

    unit_l = accuracy
    unit_p = np.array([2 * np.pi / len(i) for i in t0])
    pdiv = np.array([len(i) for i in t0])
    unit_a = 2 * np.pi / adiv

    t0[0][0][0] = 0  # this is the starting point
    q.put((0, 0, 0))

    pbar = tqdm(total=int(round(((ldiv + 1) ** 2 * adiv * np.pi))))  # show progressing bar
    while not q.empty():
        pbar.update(1)  # update progressing bar

        r, p, a = q.get()  # 0 <= r <= ldiv; 0 <= p < round(2*pi*r); 0 <= a < adiv.
        
        for SL in np.arange(V_max * dt, 0, - unit_l / 2):
            for SR in np.arange(V_max * dt, 0, - unit_l / 2):
                ds = (SL + SR) / 2
                da = int(round((SL - SR) / (2 * l * unit_a)))
                dx = ds * math.cos(a * unit_a) / unit_l
                dy = ds * math.sin(a * unit_a) / unit_l

                newR = int(round(math.sqrt((r + dx) ** 2 + dy ** 2)))
                if newR < 0 or newR > ldiv:
                    continue

                newP = (int(round((p * unit_p[r] + math.atan2(dy, r + dx)) / unit_p[newR])) + pdiv[newR]) % pdiv[newR]

                newA = (a + da + adiv) % adiv

                if t0[newR][newP][newA] > t0[r][p][a] + 1:
                    t0[newR][newP][newA] = t0[r][p][a] + 1
                    last[str(position(newR, newP, newA))] = str(position(r, p, a))
                    q.put((newR, newP, newA))

    pbar.close()

    # save json to file
    with open(dp_result_file, 'w') as ofile:
        json.dump(t0.tolist(), ofile)
    with open(dp_traceback_file, 'w') as otb:
        json.dump(last, otb)
    
    return t0, last


def meshing(t0, n):
    '''
    Args:
        t0: np.array(3d) - t[r][phi][theta]
        n: float (suggest int) - expansion rate
    Returns:
        t1: np.array(3d) - a smaller matrix
    '''
    i = 0
    t1 = []
    while int(round(n * i)) <= len(t0):
        if i == 0:
            t1i = t0[0]
        else:
            t1i = []
            pdiv = int(round(2 * np.pi * i))
            factor = len(t0[int(round(n * i))]) / pdiv
            for j in range(pdiv):
                t1i.append(t0[int(round(n * i))][int(round(j * factor))])
        t1.append(t1i)

        i += 1
    return np.array(t1)


def build_level_k(t0, ldiv, unavaliable):
    adiv = len(t0[0][0])

    # initialize time to every state to 1 hour, which is longer than any possible result
    # t1[a][r][p] (theta, r, phi)
    t1 = np.full((adiv, ldiv + 1), None)
    for t1_a in t0:
        t1_a[0] = np.array([unavaliable])
        for k in range(1, ldiv + 1):
            t1_a[k] = np.full(int(round(2 * np.pi * k)), unavaliable)
    
    # now t0[r][p][a] as before
    t1 = t1.transpose(1, 2, 0)

    unit_p1 = np.array([2 * np.pi / len(i) for i in t1])
    pdiv1 = np.array([len(i) for i in t1])
    unit_p0 = np.array([2 * np.pi / len(i) for i in t0])
    unit_a = 2 * np.pi / adiv

    q = queue.Queue()
    last = {}
    grid = {}  # this dictionary records the method (dr, dp, da) / factors for t0 used to reach a state

    t1[0][0][0] = 0
    q.put((0, 0, 0))

    pbar = tqdm(total=int(round(((ldiv + 1) ** 2 * adiv * np.pi))))
    while not q.empty():
        pbar.update(1)

        r, p, a = q.get()

        for i in range(len(t0)):
            for j in range(len(t0[i])):
                for k in range(len(t0[i][j])):
                    if t0[i][j][k] == unavaliable:
                        continue
                    rp = p * unit_p1[r]  # real p
                    ra = a * unit_a
                    dr = i  # change of r
                    dp = j * unit_p0[i]
                    da = k * unit_a
                    nx = r * np.cos(rp) + dr * np.cos(ra + dp)
                    ny = r * np.sin(rp) + dr * np.sin(ra + dp)
                    nr = math.sqrt(nx ** 2 + ny ** 2)  # new r
                    np = math.atan2(ny, nx)
                    na = ra + da

                    nrq = int(round(nr))  # new r quantized
                    if nrq >= len(t1):
                        continue

                    npq = (int(round(np / unit_p1[nrq])) + pdiv1[nrq]) % pdiv1[nrq]
                    naq = (int(round(na)) + adiv) % adiv

                    if t1[nrq][npq][naq] > t1[r][p][a] + t0[i][j][k]:
                        t1[nrq][npq][naq] = t1[r][p][a] + t0[i][j][k]
                        last[str(position(nrq, npq, naq))] = str(position(r, p, a))
                        grid[str(position(nrq, npq, naq))] = str(position(i, j, k))
                        q.put((nrq, npq, naq))

    pbar.close()

    return t1, last, grid


def traceback(last, grid, point, beginning, adiv):
    '''
    Args:
        last, grid: the dict
        point: position object
        beginning: position object - excursion of origin
        adiv: int
    Return:
        a list of the path (not including beginning)
    '''
    stack = queue.LifoQueue()
    grids = queue.LifoQueue()
    while not point.is_zero():
        stack.put(point.val())
        grids.put(position(grid[str(point)]).val())
        point = position(last[str(point)])
    
    ans = []
    ansg = []
    r0, Qp0, Qa0 = beginning.val()  # r can have any unit, but angles are not useful now as they are quantized
    p0 = Qp0 * 2 * np.pi / int(round(2 * np.pi * r0))
    a0 = Qa0 * 2 * np.pi / adiv
    while not stack.empty():
        r, Qp, Qa = stack.get()
        p = Qp * 2 * np.pi / int(round(2 * np.pi * r))
        a = Qa * 2 * np.pi / adiv

        x = r0 * np.cos(p0) + r * np.cos(a0 + p)
        y = r0 * np.sin(p0) + r * np.sin(a0 + p)

        r1 = int(round(math.sqrt(x ** 2 + y ** 2)))
        p1 = int(round(math.atan2(y, x)))
        a1 = a0 + a

        ans.append((r1, p1, a1))
        ansg.append(grids.get())

    return ans, ansg


def get_higher_quality(sub_dir, level, path0, grids0, n):
    '''
    In case MLT, use cache and release memory by layer
    In this sense, instead of dfs, bfs will be used
    Args:
        level: int - the level to be searched (level can be the max level, just make path0 a list of size 1)
        path0: a list of tuples - the path generated from the level above (which is rough) (doesn't have (0, 0, 0))
        grids0: also a list of tuples - the grid value of each state in the path
        n: the expansion
    Return:
        new path and grid
    '''
    dp_result_file = os.path.join(sub_dir, 't%d.json' % level)
    dp_traceback_file = os.path.join(sub_dir, 'trba%d.json' % level)
    dp_method_file = os.path.join(sub_dir, 'mthd%d.json' % level)

    with open(dp_result_file, 'r') as res:
        t = np.array(json.load(res))
    with open(dp_traceback_file, 'r') as tb:
        last = dict(json.load(tb))
    with open(dp_method_file, 'r') as me:
        grid = dict(json.load(me))

    assert(len(path0) == len(grids0) or print(len(path0) - len(grids0)) != None)

    path = []  # this is the new path
    grids = []  # this is the new grids
    for i in range(len(path0)):
        # positions in the last measuring level
        r0_up, p0_up, a0 = path0[i]

        # r0, p0, a0 are the positions in this measuring level (a0 doesn't need to change)
        r0 = int(round(n * r0_up))
        factor = int(round(2 * np.pi * r0)) / int(round(2 * np.pi * r0_up))
        p0 = int(round(p0_up * factor))

        # grid r, p, a of last level
        gr, gp, ga = grids0[i]

        pathi, gridi = traceback(last, grid, position(gr, gp, ga), position(r0, p0, a0), len(t[0][0]))

        path.append(pathi)
        grids.append(gridi)
    
    return path, grids


def move_origin(coords, origin):
    '''
    Args:
        coords: list of tuple of float * 3 - a list of coordinates (polar) and bearing
        origin: tuple of float * 3 - cooridinate of the origin in the new system
    '''
    if origin == (0, 0, 0):
        return coords

    r0, p0, a0 = origin

    new_coord = []
    for coord in coords:
        dr, dp, da = coord

        x = r0 * np.cos(p0) + dr * np.cos(a0 + dp)
        y = r0 * np.sin(p0) + dr * np.sin(a0 + dp)
        r = math.sqrt(x ** 2 + y ** 2)
        p = math.atan2(y, x)
        a = a0 + da

        new_coord.append((r, p, a))
    
    return new_coord


def find_path(sub_dir, start, end, coordinate, return_type, **V0):
    '''
    Args:
        start, end: All tuple(float, float, float)
        polar range (coordinate='p'): (free range, [0, 2pi), [0, 2pi))
        cartesian range (coordinate='c'): (free range, free range, [0, 2pi))
    Return:
        a list of positions (in mm)(tuple of three float), same coordinate system as input (return_type='p')
        a list of left&right diplacements (in mm) in every dt (return_type='s')
    '''
    # load argument list
    dp_argument_file = os.path.join(sub_dir, 'dp_args.json')
    with open(dp_argument_file, 'r') as arg:
        argument_list = dict(json.load(arg))
    accuracy = float(argument_list['accuracy'])
    ldiv = int(argument_list['ldiv'])
    expansion = float(argument_list['expansion'])
    adiv = int(argument_list['adiv'])

    # coordinate transfer
    if coordinate == 'p':
        x = end[0] * np.cos(end[1]) - start[0] * np.cos(start[1])
        y = end[0] * np.sin(end[1]) - start[0] * np.sin(start[1])
    elif coordinate == 'c':
        x = end[0] - start[0]
        y = end[1] - start[1]
    else:
        raise ValueError('Argument \'coordinate\' can only be \'p\' or \'c\'.')
    r = math.sqrt(x ** 2 + y ** 2)
    p = math.atan2(y, x) - start[2]
    a = end[2] - start[2]

    full_path = []
    start = (0, 0, 0)
    while r > accuracy:
        # find the layer needed
        level = max(0, math.ceil(math.log(r / (accuracy * ldiv), expansion)))

        # quantize coordinate into the layer above top layer
        rq = math.floor(r / (accuracy * expansion ** (level + 1)))
        
        # if rq is zero, means ldiv is too small (close to expansion)
        # an endless loop will be the result if this happens
        assert(rq > 0 or print('ldiv ~ expansion error') != None)

        pdiv = int(round(2 * np.pi * rq))
        pq = int(round(p / (2 *np.pi) * pdiv))
        while pq < 0:
            pq += pdiv
        while pq >= pdiv:
            pq -= pdiv

        aq = int(round(a / (2 * np.pi) * adiv))
        while aq < 0:
            a += adiv
        while aq >= adiv:
            a -= adiv

        # find path
        path = [(rq, pq, aq)]
        grid = [(0, 0, 0)]
        while level >= 0:
            path, grid = get_higher_quality(sub_dir, level, path, grid, expansion)
            level -= 1

        # calculate lefted part of r, p, q, and replace
        last_point = path[-1]
        r_done = last_point[0] * accuracy
        p_done = last_point[1] * 2 * np.pi / int(round(2 * np.pi * last_point[0]))
        a_done = last_point[2] * 2 * np.pi / adiv

        x = r * np.cos(p) - r_done * np.cos(p_done)
        y = r * np.sin(p) - r_done * np.sin(p_done)
        r = math.sqrt(x ** 2 + y ** 2)
        p = math.atan2(y, x)
        a -= a_done

        # log path
        full_path.extend(move_origin(path, start))
        start = (r_done, p_done, a_done)

    if return_type == 'p':
        return full_path
    elif return_type == 's':
        r0 = 0
        p0 = 0
        a0 = 0
        s_path = []
        for rpa in full_path:
            dx = rpa[0] * np.cos(rpa[1]) - r0 * np.cos(p0)
            dy = rpa[0] * np.sin(rpa[1]) - r0 * np.sin(p0)
            ds = math.sqrt(dx ** 2 + dy ** 2)
            dal = (rpa[2] - a0) * int(argument_list['l'])
            r0, p0, a0 = rpa
            SL = ds + dal
            SR = ds - dal
            s_path.append((SL, SR))
        return s_path
    else:
        print('not supported return type')
        print('return default type (p-type)')
        return full_path


def store(sub_dir, level, t, last, grid):
    '''
    Args:
        sub_dir: string
        level: int
        t: numpy.ndarray
        last: dict
        grid: dict
    '''
    dp_result_file = os.path.join(sub_dir, 't%d.json' % level)
    dp_traceback_file = os.path.join(sub_dir, 'trba%d.json' % level)
    dp_method_file = os.path.join(sub_dir, 'mthd%d.json' % level)

    with open(dp_result_file, 'w') as ofile:
        json.dump(t.tolist(), ofile)
    with open(dp_traceback_file, 'w') as otb:
        json.dump(last, otb)
    with open(dp_method_file, 'w') as omt:
        json.dump(grid, omt)


def build_dp_result_matrix(V_max, l, ldiv=40, adiv=360, accuracy=1.0, expansion=5.0, dt=0.1, L_max=3000):
    '''
    Args:
        V_max: float - max velocity a wheel can reach (in mm/s);
        A_max: float - max acceration a wheel can reach (in mm/s^2);
        l: float - the abeam distance between wheel and the centre (in mm);
        ldiv: unsigned int - discribes the accuracy of dividing length each time (larger = more precise);
        accuracy: the overall accuracy want to achieve (in mm);
        expansion: the expansion factor of the map each time (can be float);
        adiv: unsigned int - discribes the accuracy of dividing angle / bearing (larger = more precise);
        dt: float - the minimum unit of time (in s);
        L_max: float - size of the field (in mm);
        VL0, VR0: float - the speed of left-side and right-side wheels at the beginning
    Notice:
        time complexity: O()
        space complexity: O()
    Returns:
        Store data, return max level
    Boundness:
        To compromise space, only suitable for large acceleration.
    '''
    # build a sub-directory for storage (if it havn't been built before)
    sub_dir = 'cache3'
    try:
        os.makedirs(sub_dir)
    except FileExistsError:
        pass
    dp_argument_file = os.path.join(sub_dir, 'dp_args.json')

    unavaliable = int(3600 / dt)
    current_argument_list = {'V_max': V_max, 'l': l, 'ldiv': ldiv, 'adiv': adiv, 'dt': dt, 'L_max': L_max, 
                             'accuracy': accuracy, 'expansion': expansion, 'unavaliable': unavaliable}
    
    try:
        with open(dp_argument_file, 'r') as arg:
            argument_list = dict(json.load(arg))
        assert(argument_list == current_argument_list)  # this judgement will be imporved later, to satisfy the need of slight change
        return
    except:
        pass

    # build the first level
    t0, last0 = build_level_0(sub_dir, V_max, l, ldiv, adiv, accuracy, dt, unavaliable)

    t = t0
    last = last0
    grid = None  # the number of the grid used in each movement
    mesh = None  # the mesh USED BY this level (NOT generated from this level)
    level = 0

    # in terms of storage, first layer will store the data within the function, while other levels through this function
    while accuracy * expansion ** level * ldiv < L_max:
        level += 1
        mesh = meshing(t, expansion)
        t, last, grid = build_level_k(mesh, ldiv, unavaliable)
        store(sub_dir, level, t, last, grid)

    with open(dp_argument_file, 'w') as oarg:
        json.dump(current_argument_list, oarg)

