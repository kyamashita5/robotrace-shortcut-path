import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import time

from util import *
from smooth_path import smooth_path


#pts = np.loadtxt('./data/robotena_points.txt', delimiter=' ')
#pts = np.loadtxt('./data/2019kansai_points.txt', delimiter=' ')
pts = np.loadtxt('./data/2015alljapan_points.txt', delimiter=' ')


# initialize nodes
nodes = [pts, smooth_path(pts)]
#for dn in [-0.05, -0.025, 0.025, 0.05]:
#    pts_offset = np.zeros_like(pts)
#    for idx, pt in enumerate(pts):
#        ang = course_angle(pts, idx)
#        ofs = np.array([-np.sin(ang), np.cos(ang)])
#        pts_offset[idx] = pts[idx] + dn * ofs
#    nodes.append(pts_offset)
nodes = np.stack(nodes, axis=1) # [N,K,2]

for k in range(nodes.shape[1]):
    plt.plot(nodes[:,k,0], nodes[:,k,1])
plt.axes().set_aspect('equal')
plt.grid()
plt.show()

def get_adjacent(idx_x, idx_y, nodes, dl = 0.01, margin = 0.05, ang_th = np.pi/180.0, curv_lim = 10.0):
    ang = course_angle(nodes[:,idx_y], idx_x)
    pt0 = nodes[idx_x,idx_y]
    list_adj = []

    if ((idx_x + 1) < len(nodes)) and (idx_y == 0):
        list_adj.append([2*dl, [idx_x + 1, 0], [idx_x, idx_y]])

    idx_i = idx_x + 1
    cmin = -10.0
    cmax = 10.0
    while (idx_i < len(nodes)):
        # 打ち切り判定
        ang1 = course_angle(nodes[:,0], idx_i)
        pt_l = nodes[idx_i,0] + margin * np.array([-np.sin(ang1), np.cos(ang1)])
        dif_t_l = np.dot(np.array([np.cos(ang), np.sin(ang)]), pt_l-pt0)
        dif_n_l = np.dot(np.array([-np.sin(ang), np.cos(ang)]), pt_l-pt0)
        cmax = min([cmax, 2*dif_n_l/(dif_t_l**2+dif_n_l**2)])
        pt_r = nodes[idx_i,0] - margin * np.array([-np.sin(ang1), np.cos(ang1)])
        dif_t_r = np.dot(np.array([np.cos(ang), np.sin(ang)]), pt_r-pt0)
        dif_n_r = np.dot(np.array([-np.sin(ang), np.cos(ang)]), pt_r-pt0)
        cmin = max([cmin, 2*dif_n_r/(dif_t_r**2+dif_n_r**2)])
        if cmin >= cmax:
            break

        for idx_j in range(nodes.shape[1]):
            ang1 = course_angle(nodes[:,idx_j], idx_i)
            dif_t = np.dot(np.array([ np.cos(ang), np.sin(ang)]), nodes[idx_i, idx_j] - pt0)
            dif_n = np.dot(np.array([-np.sin(ang), np.cos(ang)]), nodes[idx_i, idx_j] - pt0)
            c = 2 * dif_n / (dif_t * dif_t + dif_n * dif_n)
            ang_radius = 2.0 * np.arctan2(np.abs(dif_n), dif_t)
            ang1_ = ang + (ang_radius if (c > 0.0) else -ang_radius)
            err_ang = np.abs(np.mod(ang1 - ang1_ + np.pi, 2.0  * np.pi) - np.pi)

            if (err_ang < ang_th) and (cmin < c) and (cmax > c): # and (idx_i >= idx_min)
                # straight
                if (ang_radius < (np.pi / 180.0)):
                    l = np.sqrt(dif_t * dif_t + dif_n * dif_n)
                    list_adj.append([cost_straight(l), [idx_i, idx_j], [idx_x, idx_y]])
                # turn
                elif np.abs(c) < curv_lim:
                    l = np.sqrt(dif_t * dif_t + dif_n * dif_n) / np.sinc(0.5 * ang_radius / np.pi)
                    list_adj.append([cost_turn(l, c), [idx_i, idx_j], [idx_x, idx_y]])
            
            # TODO: 枝刈り
            
        idx_i += 1
    return list_adj

def draw_arch(node0, node1, nodes, dl = 0.001, ang_th = np.pi/180.0):
    ang0 = course_angle(nodes[:,node0[1]], node0[0])
    pt0 = nodes[node0[0], node0[1]]
    ang1 = course_angle(nodes[:,node1[1]], node1[0])
    pt1 = nodes[node1[0], node1[1]]
    dif_ang = ang1 - ang0
    dif_ang = (dif_ang - 2.0 * np.pi) if (dif_ang > np.pi) else dif_ang
    dif_ang = (dif_ang + 2.0 * np.pi) if (dif_ang < -np.pi) else dif_ang    

    dif_t = np.cos(ang0) * (pt1[0] - pt0[0]) + np.sin(ang0) * (pt1[1] - pt0[1])
    dif_n = -np.sin(ang0) * (pt1[0] - pt0[0]) + np.cos(ang0) * (pt1[1] - pt0[1])
    c = 2 * dif_n / (dif_t*dif_t + dif_n*dif_n)
    ang_radius = 2.0 * np.arctan2(np.abs(dif_n), dif_t)
    l = np.sqrt(dif_t*dif_t + dif_n*dif_n) / np.sinc(0.5 * ang_radius / np.pi)
    if ang_radius < ang_th:
        # 直進
        plt.plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], color='m')
    else:
        # 円弧
        t0 = np.array([np.cos(ang0), np.sin(ang0)])
        n0 = np.array([-np.sin(ang0), np.cos(ang0)])
        x_draw = []
        y_draw = []
        for s in np.arange(0, l, dl):
            dt = s * np.sinc(c * s / np.pi)
            dn = s * np.sinc(0.5 * c * s / np.pi) * np.sin(0.5 * c * s)
            pt_draw = pt0 + dt * t0 + dn * n0
            x_draw.append(pt_draw[0])
            y_draw.append(pt_draw[1])
        x_draw.append(pt1[0])
        y_draw.append(pt1[1])
        plt.plot(x_draw, y_draw, color='y')

dl = 0.01

idx_x0 = 0
idx_y0 = 0
list_adj = get_adjacent(idx_x0, idx_y0, nodes, dl)
plt.plot(nodes[:,0,0], nodes[:,0,1])
for adj in list_adj:
    draw_arch(adj[2], adj[1], nodes, dl)
print('num of archs : ' + str(len(list_adj)))
plt.axes().set_aspect('equal')
plt.grid()
plt.show()


print('start computing shortest path')
start = time.time()
costs = -1*np.ones_like(nodes[:,:,0])
decided = np.zeros_like(nodes[:,:,0])
srcs = -1*np.ones_like(nodes, dtype='int')
h = []
# 1. push start node into heapq
costs[0,0] = 0.0
heappush(h, [0, [0,0]])
tmp_idx = 0
while len(h) > 0:
    print(str(tmp_idx) + '/' + str(len(nodes)) + ', ' + str(len(h)))
    val = heappop(h)
    cost = val[0]
    node = val[1]
    if decided[node[0], node[1]]:
        continue
    tmp_idx = max([tmp_idx, node[0]])
    decided[node[0], node[1]] = True
    if (node[0] == (len(nodes)-1)) and (node[1] == 0):
        break
    list_adj = get_adjacent(node[0], node[1], nodes, dl)
    for adj in list_adj:
        cost_dif = adj[0]
        dst = adj[1]
        if (costs[dst[0], dst[1]] < 0) or ((cost+cost_dif) < costs[dst[0], dst[1]]):
            costs[dst[0], dst[1]] = cost+cost_dif
            srcs[dst[0], dst[1]] = node
            heappush(h, [cost+cost_dif, [dst[0], dst[1]]])
elapsed = time.time() - start
print('finished')
print('elapsed : ' + str(elapsed))
print('cost: ', costs[-1,0])

# draw result
plt.plot(nodes[:,0,0], nodes[:,0,1])
dst = [len(nodes)-1, 0]
while True:
    src = srcs[dst[0], dst[1]]
    draw_arch(src, dst, nodes, dl)
    #print(src, dst)
    if src[0] == 0 and src[1] == 0:
        break
    dst = src
plt.axes().set_aspect('equal')
plt.grid()
plt.show()
