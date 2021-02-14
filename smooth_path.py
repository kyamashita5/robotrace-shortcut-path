import numpy as np
import matplotlib.pyplot as plt

# 角度が急激に変化するところでは移動平均の窓を小さくする

def curvature(pts, idx, m = 5):
    if m > min(m, idx, len(pts)-idx-1):
        return 0.0
    v1 = pts[idx] - pts[idx-m]
    v2 = pts[idx+m] - pts[idx]
    dif_ang = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    dif_len = 0.5*(np.linalg.norm(v1)+np.linalg.norm(v2))
    if dif_ang < -np.pi:
        dif_ang += 2.0 * np.pi
    elif dif_ang > np.pi:
        dif_ang -= 2.0 * np.pi
    return dif_ang / dif_len

def angle(pts, idx, m = 5):
    if (idx - m) < 0:
        return angle(pts, m, m)
    elif (idx + m) >= len(pts):
        return angle(pts, len(pts)-m-1, m)
    v = pts[idx+m] - pts[idx-m]
    return np.arctan2(v[1],v[0])

def smooth_path(pts):
    route = pts.copy()

    window_size = []

    for i in range(len(pts)):
        m = 10
        ang_diff = angle(pts, i+m) - angle(pts, i-m)
        while ang_diff > np.pi:
            ang_diff -= 2.0*np.pi
        while ang_diff < -np.pi:
            ang_diff += 2.0*np.pi
        if np.abs(ang_diff) > 0.6*np.pi:
            window_size.append(10)
        else:
            window_size.append(20)
    window_size = np.array(window_size)

    cnt = 0
    for i in range(len(pts)):
        if window_size[i] == 10:
            cnt = 15
        if cnt > 0:
            window_size[i] = 10
            cnt -= 1
    for i in range(len(pts)):
        if window_size[len(pts)-1-i] == 10:
            cnt = 15
        if cnt > 0:
            window_size[len(pts)-1-i] = 10
            cnt -= 1

    for i in range(2, len(pts)):
        n_lim = int(0.5*(window_size[i-2] + window_size[i-1])) + 1
        window_size[i] = min([window_size[i], n_lim])
    for i in range(2, len(pts)):
        n_lim = int(0.5*(window_size[len(pts)-1-i+2] + window_size[len(pts)-1-i+1])) + 1
        window_size[len(pts)-1-i] = min([window_size[len(pts)-1-i], n_lim])
    #plt.plot(window_size)
    #plt.show()

    #plt.plot(window_size[1:]-window_size[:-1])
    #plt.show()

    for i in range(len(pts)):
        n = min([window_size[i], i, len(pts)-1-i])
        idx_s = max([0,i-n])
        idx_e = min([len(pts),i+n+1])
        route[i] = np.mean(pts[idx_s:idx_e], axis=0)

    return route

if __name__ == '__main__':
    linedata = np.loadtxt('./data/2019kansai_points.txt')
    pts_l = linedata[:,0:2]

    route = smooth_path(pts_l)

    curv_l = []
    curv_r = []
    for i in range(len(pts_l)):
        curv_l.append(curvature(pts_l, i))

    for i in range(len(route)):
        curv_r.append(curvature(route, i))

    plt.plot(pts_l[:,0], pts_l[:,1])
    plt.plot(route[:,0],route[:,1])
    plt.show()

    plt.plot(curv_l)
    plt.plot(curv_r)
    plt.ylim((-20,20))
    plt.grid()
    plt.show()