import numpy as np

def arith_div(x, y):
    if x >= 0.0:
        return int(x/y)
    else:
        return int(x/y)-1

def course_angle(pts, idx, m = 5):
    if idx < m:
        return course_angle(pts, m,m=m)
    elif idx > (len(pts) - m - 1):
        return course_angle(pts, len(pts)-1-m, m=m)
    m = min(m, idx, len(pts)-idx-1)
    dx = pts[idx+m, 0] - pts[idx-m, 0]
    dy = pts[idx+m, 1] - pts[idx-m, 1]    
    return np.arctan2(dy,dx)

def cost_straight(l, acc = 9.0, vmax = 4.0, vturn = 1.0):
    lth = (vmax * vmax - vturn * vturn) / acc
    if (l < lth):
        t = 2.0 * (np.sqrt(vturn * vturn + l * acc) - vturn) / acc
    else: 
        t = (l + (vmax - vturn) * (vmax - vturn) / acc) / vmax    
    return t

def cost_lanechange(dif_t, dif_n, acc = 6.0, vmax = 3.0, vturn = 1.0, cf = 10.0):
    dif_n = np.abs(dif_n)
    c = 4*dif_n/(dif_t**2+dif_n**2)
    if c > 1.0:
        ang = np.arctan2(0.5*dif_t*c, 1.0-0.5*dif_n*c)
        l = 2.0*ang/c
    else:
        l = dif_t + dif_n**2/dif_t
    if np.abs(c) > 0.01:
        vmax = min([vmax, np.sqrt(cf/np.abs(c))])
    lth = (vmax * vmax - vturn * vturn) / acc
    if l < 0.05:
        t = l / vturn
    elif (l < lth):
        t = 2.0 * (np.sqrt(vturn * vturn + l * acc) - vturn) / acc
    else: 
        t = (l + (vmax - vturn) * (vmax - vturn) / acc) / vmax    
    return t

def cost_turn(l, c, acc = 6.0, vmax = 3.0, vturn = 1.0, cf = 10.0):
    vmax = min([vmax, np.sqrt(cf/np.abs(c))])
    lth = (vmax * vmax - vturn * vturn) / acc
    if l < 0.05:
        t = l / vturn
    elif (l < lth):
        t = 2.0 * (np.sqrt(vturn * vturn + l * acc) - vturn) / acc
    else: 
        t = (l + (vmax - vturn) * (vmax - vturn) / acc) / vmax    
    return t


