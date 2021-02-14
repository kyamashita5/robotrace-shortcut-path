import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import time
#from numba import jit


# int(x / y)
#@jit
def arith_div(x, y):
    if x >= 0.0:
        return int(x/y)
    else:
        return int(x/y)-1


#data = np.loadtxt('map_2017.txt', delimiter=',')
#pts = data[:, 1:3]

#t = np.arange(0,60,0.01)
#x = t.copy()
#y = 0*t.copy()
#pts = np.array([x,y]).T

#pts = np.loadtxt('./data/robotena_points.txt', delimiter=' ')
pts = np.loadtxt('./data/2019kansai_points.txt', delimiter=' ')
#pts = np.loadtxt('./data/2018alljapan_points.txt', delimiter=' ')

#data_m = np.loadtxt('marker.txt', delimiter=',')
#pts_m = data_m[:,1:3]
#ang_m = data_m[:,3]

#plot_marker([0,0.05], 0)

plt.plot(pts[:, 0], pts[:, 1])
#for pt, ang in zip(pts_m, ang_m):
#    plot_marker(pt, ang)
plt.show()

#@jit
def course_angle(pts, idx, m = 5):
    if idx < m:
        return course_angle(pts, m,m=m)
    elif idx > (len(pts) - m - 1):
        return course_angle(pts, len(pts)-1-m, m=m)
    m = min(m, idx, len(pts)-idx-1)
    dx = pts[idx+m, 0] - pts[idx-m, 0]
    dy = pts[idx+m, 1] - pts[idx-m, 1]    
    return np.arctan2(dy,dx)

for idx, pt in enumerate(pts):
    ang = course_angle(pts, idx)
    pt2 = pt + 0.01*np.array([np.cos(ang), np.sin(ang)])
    plt.plot([pt[0], pt2[0]], [pt[1], pt2[1]], color='m')
plt.show()

#@jit
def cost_straight(l, acc = 9.0, vmax = 4.0, vturn = 1.0):
    lth = (vmax * vmax - vturn * vturn) / acc
    if (l < lth):
        t = 2.0 * (np.sqrt(vturn * vturn + l * acc) - vturn) / acc
    else: 
        t = (l + (vmax - vturn) * (vmax - vturn) / acc) / vmax    
    return t

#@jit
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

#@jit
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

#@jit
def get_adjcent(idx_x, idx_y, pts, dl = 0.01, lim_idx_y = 7, ang_th = np.pi/180.0, curv_lim = 10.0):
    idx_i = idx_x + 1
    ang = course_angle(pts, idx_x)
    pt0 = pts[idx_x] + dl * idx_y * np.array([-np.sin(ang), np.cos(ang)])
    list_adj = []
    if (idx_i < len(pts)) and (idx_y == 0):
        list_adj.append([2*dl, [idx_i, 0], [idx_x, idx_y]])
    cmin = -10.0
    cmax = 10.0
    while (idx_i < len(pts)):
        ang1 = course_angle(pts, idx_i)
        pt_l = pts[idx_i] + lim_idx_y * dl * np.array([-np.sin(ang1), np.cos(ang1)])
        dif_t_l = np.dot(np.array([np.cos(ang), np.sin(ang)]), pt_l-pt0)
        dif_n_l = np.dot(np.array([-np.sin(ang), np.cos(ang)]), pt_l-pt0)
        cmax = min([cmax, 2*dif_n_l/(dif_t_l**2+dif_n_l**2)])
        pt_r = pts[idx_i] - lim_idx_y * dl * np.array([-np.sin(ang1), np.cos(ang1)])
        dif_t_r = np.dot(np.array([np.cos(ang), np.sin(ang)]), pt_r-pt0)
        dif_n_r = np.dot(np.array([-np.sin(ang), np.cos(ang)]), pt_r-pt0)
        cmin = max([cmin, 2*dif_n_r/(dif_t_r**2+dif_n_r**2)])
        if cmin >= cmax:
            break
        dif_ang = ang1 - ang
        dif_ang = (dif_ang - 2.0 * np.pi) if (dif_ang > np.pi) else dif_ang
        dif_ang = (dif_ang + 2.0 * np.pi) if (dif_ang < -np.pi) else dif_ang
        # 直進
        dif_n = np.dot(np.array([-np.sin(ang), np.cos(ang)]), pts[idx_i] - pt0)
        dif_t = np.dot(np.array([ np.cos(ang), np.sin(ang)]), pts[idx_i] - pt0)
        if (cmin < 0.0) and (cmax > 0.0) and (dif_t > 0.0):            
            if np.abs(dif_ang) < ang_th:
                dif_ni = np.dot(np.array([-np.sin(ang1), np.cos(ang1)]), pt0 - pts[idx_i])
                idx_j = arith_div((dif_ni + 0.5*dl), dl)
                if (np.abs(idx_j) <= lim_idx_y):
                    list_adj.append([cost_straight(dif_t), [idx_i, idx_j], [idx_x, idx_y]])
                if dif_t > 0.3:
                    # レーンチェンジ
                    for k in [idx_j-1, idx_j+1]:
                        if np.abs(k) <= lim_idx_y:
                            list_adj.append([cost_lanechange(dif_t, dif_n), [idx_i, k], [idx_x, idx_y]])
        # 円弧(~90deg)
        valid_arch = False
        if (np.abs(dif_ang) > ang_th) and (np.abs(dif_ang) < 0.75*np.pi):
            t0 = np.array([np.cos(ang), np.sin(ang)])
            n0 = np.array([-np.sin(ang), np.cos(ang)])
            t1 = np.array([np.cos(ang1), np.sin(ang1)])
            n1 = np.array([-np.sin(ang1), np.cos(ang1)])
            pt1_cen = pts[idx_i]
            err_n = (-np.dot(t1-t0, pt1_cen-pt0) / np.dot(t1-t0, n1))
            #if dif_ang > 0.0:
            #    idx_j = 
            idx_j = arith_div((err_n+0.5*dl), dl)
            pt1 = pts[idx_i] + dl * idx_j * np.array([-np.sin(ang1), np.cos(ang1)])
            if np.abs(idx_j) < lim_idx_y:
                # 間の点のずれが許容範囲ならアーク認定
                R = np.linalg.norm(pt1-pt0) / (2.0*np.sin(0.5*dif_ang))

                valid_arch = True
                if np.abs(R) < (1.0 / curv_lim):
                    valid_arch = False
                if valid_arch:
                    curv = (1.0 / R)
                    if (curv < cmin) or (curv > cmax):
                        valid_arch = False
                if valid_arch:
                    #print(idx_i, idx_j, R)
                    len_arch = R*dif_ang
                    if len_arch > 0.0:
                        cost = cost_turn(R*dif_ang, 1.0/R)
                        list_adj.append([cost, [idx_i, idx_j], [idx_x, idx_y]])

                pass
            
        idx_i += 1
    return list_adj

def draw_arch(node0, node1, pts, dl = 0.01, ang_th = np.pi/180.0):
    ang0 = course_angle(pts, node0[0])
    pt0 = pts[node0[0]] + dl * node0[1] * np.array([-np.sin(ang0), np.cos(ang0)])
    ang1 = course_angle(pts, node1[0])
    pt1 = pts[node1[0]] + dl * node1[1] * np.array([-np.sin(ang1), np.cos(ang1)])
    dif_ang = ang1 - ang0
    dif_ang = (dif_ang - 2.0 * np.pi) if (dif_ang > np.pi) else dif_ang
    dif_ang = (dif_ang + 2.0 * np.pi) if (dif_ang < -np.pi) else dif_ang    
    if np.abs(dif_ang) < ang_th:
        err_t = np.cos(ang1) * (pt1[0] - pt0[0]) + np.sin(ang1) * (pt1[1] - pt0[1])
        err_n = -np.sin(ang1) * (pt1[0] - pt0[0]) + np.cos(ang1) * (pt1[1] - pt0[1])
        # 直進
        if np.abs(err_n) < dl:
            plt.plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], color='m')
        # レーンチェンジ
        else:
            c = 4.0*err_n / (err_t**2+err_n**2)
            ang_arch = np.arctan2(0.5*err_t*c, 1.0-0.5*err_n*c)
            t = np.array([np.cos(ang1), np.sin(ang1)])
            n = np.array([-np.sin(ang1), np.cos(ang1)])
            if ang_arch < 0.0:
                n = -1.0 * n
            draw_x = []
            draw_y = []
            for ang_tmp in np.arange(0.0, np.abs(ang_arch), np.abs(ang_arch)/50.0):
                pt_tmp = pt0 + np.sin(ang_tmp) * t + (1-np.cos(ang_tmp)) * n
                draw_x.append(pt_tmp[0])
                draw_y.append(pt_tmp[1])
            for ang_tmp in np.arange(np.abs(ang_arch), 0.0, -np.abs(ang_arch)/50):
                pt_tmp = pt1 - np.sin(ang_tmp) * t - (1-np.cos(ang_tmp)) * n
                draw_x.append(pt_tmp[0])
                draw_y.append(pt_tmp[1])
            plt.plot(draw_x, draw_y, color='g')
    else:
        # 円弧
        t0 = np.array([np.cos(ang0), np.sin(ang0)])
        n0 = np.array([-np.sin(ang0), np.cos(ang0)])
        R = np.linalg.norm(pt1-pt0) / (2.0*np.sin(0.5*dif_ang))
        x_draw = []
        y_draw = []
        for ang in np.arange(0, dif_ang, dif_ang/100.0):
            pt_draw = pt0 + R*n0 + R * (-np.cos(ang)*(n0) + np.sin(ang)*t0)
            x_draw.append(pt_draw[0])
            y_draw.append(pt_draw[1])
        pt_draw = pt0 + R*n0 + R * (-np.cos(dif_ang)*(n0) + np.sin(dif_ang)*t0)
        x_draw.append(pt_draw[0])
        y_draw.append(pt_draw[1])
        x_draw += 0.5 * (pt0[0] - x_draw[0] + pt1[0] - x_draw[-1])
        y_draw += 0.5 * (pt0[1] - y_draw[0] + pt1[1] - y_draw[-1])
        plt.plot(x_draw, y_draw, color='y')

dl = 0.01
lim_idx_y = 5

idx_x0 = 0
idx_y0 = 0
list_adj = get_adjcent(idx_x0, idx_y0, pts, dl, lim_idx_y)
plt.plot(pts[:,0], pts[:,1])
for adj in list_adj:
    draw_arch(adj[2], adj[1], pts, dl)
print('num of archs : ' + str(len(list_adj)))
plt.show()

print('start computing shortest path')
start = time.time()
costs = -1*np.ones((len(pts), 2*lim_idx_y+1))
decided = np.zeros((len(pts), 2*lim_idx_y+1))
srcs = -1*np.ones((len(pts), 2*lim_idx_y+1, 2), dtype='int')
h = []
# 1. push start node into heapq
heappush(h, [0, [0,0]])
tmp_idx = 0
while len(h) > 0:
    print(str(tmp_idx) + '/' + str(len(pts)) + ', ' + str(len(h)))
    val = heappop(h)
    cost = val[0]
    node = val[1]
    if decided[node[0], node[1]+lim_idx_y]:
        continue
    tmp_idx = max([tmp_idx, node[0]])
    decided[node[0], node[1]+lim_idx_y] = True
    if (node[0] == (len(pts)-1)) and (node[1] == 0):
        break
    list_adj = get_adjcent(node[0], node[1], pts, dl, lim_idx_y)
    for adj in list_adj:
        cost_dif = adj[0]
        dst = adj[1]
        if (costs[dst[0], dst[1]+lim_idx_y] < 0) or ((cost+cost_dif) < costs[dst[0], dst[1]+lim_idx_y]):
            costs[dst[0], dst[1]+lim_idx_y] = cost+cost_dif
            srcs[dst[0], dst[1]+lim_idx_y] = node
            heappush(h, [cost+cost_dif, [dst[0], dst[1]]])
elapsed = time.time() - start
print('finished')
print('elapsed : ' + str(elapsed))

# draw res
plt.plot(pts[:,0], pts[:,1])
dst = [len(pts)-1, 0]
while True:
    src = srcs[dst[0], dst[1]+lim_idx_y]
    draw_arch(src, dst, pts, dl)
    #print(src, dst)
    if src[0] == 0 and src[1] == 0:
        break
    dst = src
plt.grid()
plt.show()
