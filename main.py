import numpy as np
import cv2
import sys
import imutils
from pylsd2 import lsd
from random import randint
from copy import deepcopy

process_stage = 0
prev_stage = -1
ix, iy = -1, -1
flagx = 0
edit_flag = 0
boxes = []

def cross2d(a, b):
    return a[0]*b[1] - a[1]*b[0]

def thinning_iteration(im, iteration):
    I = im.copy()
    M = np.zeros_like(I)
    rows, cols = I.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            p2 = I[i-1, j]
            p3 = I[i-1, j+1]
            p4 = I[i, j+1]
            p5 = I[i+1, j+1]
            p6 = I[i+1, j]
            p7 = I[i+1, j-1]
            p8 = I[i, j-1]
            p9 = I[i-1, j-1]
            A = (
                (p2 == 0 and p3 == 1) +
                (p3 == 0 and p4 == 1) +
                (p4 == 0 and p5 == 1) +
                (p5 == 0 and p6 == 1) +
                (p6 == 0 and p7 == 1) +
                (p7 == 0 and p8 == 1) +
                (p8 == 0 and p9 == 1) +
                (p9 == 0 and p2 == 1)
            )
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            if iteration == 0:
                m1 = p2 * p4 * p6
                m2 = p4 * p6 * p8
            else:
                m1 = p2 * p4 * p8
                m2 = p2 * p6 * p8
            if A == 1 and 2 <= B <= 6 and m1 == 0 and m2 == 0:
                M[i, j] = 1
    return I & ~M

def thinning(src):
    bin_ = (src.copy() // 255).astype(np.uint8)
    prev = np.zeros_like(bin_)
    while True:
        tmp = thinning_iteration(bin_, 0)
        bin_ = thinning_iteration(tmp, 1)
        diff = np.abs(bin_ - prev)
        if np.sum(diff) == 0:
            break
        prev = bin_.copy()
    return (bin_ * 255).astype(np.uint8)

def skeleton_points(skel):
    sk = skel.copy()
    sk[sk != 0] = 1
    sk = sk.astype(np.uint8)
    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]], dtype=np.uint8)
    f = cv2.filter2D(sk, -1, kernel)
    ends = np.where(f == 11)
    return ends

def point_to_line_dist(pt, line2d):
    vec = line2d[1] - line2d[0]
    norm_ = np.linalg.norm(vec) + 1e-12
    rel = line2d[0] - pt
    crossv = abs(cross2d(vec, rel))
    dist_line = crossv / norm_
    t = ((pt[0]-line2d[0][0])*vec[0] + (pt[1]-line2d[0][1])*vec[1]) / (norm_**2)
    ix = line2d[0][0] + t*vec[0]
    iy = line2d[0][1] + t*vec[1]
    d_end = min(np.linalg.norm(pt - line2d[0]),
                np.linalg.norm(pt - line2d[1]))
    minx, maxx = sorted([line2d[0][0], line2d[1][0]])
    miny, maxy = sorted([line2d[0][1], line2d[1][1]])
    if (minx <= ix <= maxx) and (miny <= iy <= maxy):
        return dist_line
    else:
        return d_end

def lines_between_ends(ends):
    v_pairs, h_pairs = [], []
    skel = cv2.imread("data/skel.pgm", 0)
    if skel is None:
        return np.zeros((0,4), dtype=np.float32), np.zeros((0,4), dtype=np.float32)
    for i in range(ends[0].size):
        y1 = ends[0][i]
        x1 = ends[1][i]
        for j in range(i+1, ends[0].size):
            y2 = ends[0][j]
            x2 = ends[1][j]
            dist_ = np.hypot(x1 - x2, y1 - y2)
            if dist_ < 5 or dist_ > 100:
                continue
            ang_ = abs(np.rad2deg(np.arctan2(y1 - y2, x1 - x2)))
            mnx, mxx = sorted([x1, x2])
            mny, mxy = sorted([y1, y2])
            if 75 < ang_ < 105:
                sub = skel[max(mny-5,0):mxy+5, max(mnx-5,0):mxx+5]
                q_ = np.where(sub == 255)
                ccount = 0
                for k_ in range(len(q_[0])):
                    px = (mnx - 5) + q_[1][k_]
                    py = (mny - 5) + q_[0][k_]
                    dd = point_to_line_dist(np.array([px, py]),
                                            np.array([[x1, y1],
                                                      [x2, y2]], dtype=np.float32))
                    if dd < 2:
                        ccount += 1
                if ccount > dist_ * 0.7:
                    v_pairs.append([x1, y1, x2, y2])
            elif ang_ < 15 or ang_ > 170:
                sub = skel[max(mny-5,0):mxy+5, max(mnx-5,0):mxx+5]
                q_ = np.where(sub == 255)
                ccount = 0
                for k_ in range(len(q_[0])):
                    px = (mnx - 5) + q_[1][k_]
                    py = (mny - 5) + q_[0][k_]
                    dd = point_to_line_dist(np.array([px, py]),
                                            np.array([[x1, y1],
                                                      [x2, y2]], dtype=np.float32))
                    if dd < 2:
                        ccount += 1
                if ccount > dist_ * 0.7:
                    h_pairs.append([x1, y1, x2, y2])
    v_pairs = np.array(v_pairs, dtype=np.float32).reshape(-1, 4)
    h_pairs = np.array(h_pairs, dtype=np.float32).reshape(-1, 4)
    return v_pairs, h_pairs

def box_between_ends(lines):
    out = []
    if len(lines) == 0:
        return out
    n_ = len(lines)
    lookup = np.full((n_, n_), -1, dtype=int)
    for i in range(n_):
        for j in range(i+1, n_):
            mx1 = (lines[i,0] + lines[i,2]) / 2
            my1 = (lines[i,1] + lines[i,3]) / 2
            mx2 = (lines[j,0] + lines[j,2]) / 2
            my2 = (lines[j,1] + lines[j,3]) / 2
            dd_ = np.hypot(mx1 - mx2, my1 - my2)
            if dd_ < 60:
                c_ = np.where(lookup == j)
                if len(c_[0]) == 0:
                    cx_ = np.where(lookup == i)
                    if len(cx_[0]) == 0:
                        lookup[i, j] = j
                    else:
                        lookup[cx_[0][0], j] = j
                else:
                    lookup[c_[0][0], i] = i
    for k in range(lookup.shape[0]):
        c_ = np.where(lookup[k, :] != -1)
        if len(c_[0]) == 1:
            i, j = k, c_[0][0]
            pts = np.array([
                [lines[i,0], lines[i,1]],
                [lines[i,2], lines[i,3]],
                [lines[j,0], lines[j,1]],
                [lines[j,2], lines[j,3]]
            ], dtype=np.float32)
            mnx, mxx = int(pts[:,0].min()) - 5, int(pts[:,0].max()) + 5
            mny, mxy = int(pts[:,1].min()) - 5, int(pts[:,1].max()) + 5
            h1 = np.hypot(lines[i,0] - lines[i,2], lines[i,1] - lines[i,3])
            h2 = np.hypot(lines[j,0] - lines[j,2], lines[j,1] - lines[j,3])
            big, sm = max(h1, h2), min(h1, h2)
            if sm < 1e-9:
                continue
            ratio = big / sm
            if ratio <= 1.2:
                idx_ = 1
            else:
                idx_ = 0
            w_ = mxx - mnx
            h_ = mxy - mny
            out.append([(mnx, mny, w_, h_), idx_])
        elif len(c_[0]) == 2:
            x_ = k
            y_, z_ = c_[0][0], c_[0][1]
            pts = np.array([
                [lines[x_,0], lines[x_,1]],
                [lines[x_,2], lines[x_,3]],
                [lines[y_,0], lines[y_,1]],
                [lines[y_,2], lines[y_,3]],
                [lines[z_,0], lines[z_,1]],
                [lines[z_,2], lines[z_,3]]
            ], dtype=np.float32)
            mnx, mxx = int(pts[:,0].min()) - 10, int(pts[:,0].max()) + 10
            mny, mxy = int(pts[:,1].min()) - 10, int(pts[:,1].max()) + 15
            w_ = mxx - mnx
            h_ = mxy - mny
            out.append([(mnx, mny, w_, h_), 2])
    return out

def remove_parallel_lines(lsd_lines, axis):
    lines = []
    if axis == 2:
        for ln in lsd_lines:
            x1, y1, x2, y2, w, *_ = ln
            a_ = abs(np.rad2deg(np.arctan2(y1 - y2, x1 - x2)))
            if 80 <= a_ <= 100:
                lines.append(ln)
    else:
        for ln in lsd_lines:
            x1, y1, x2, y2, w, *_ = ln
            a_ = abs(np.rad2deg(np.arctan2(y1 - y2, x1 - x2)))
            if a_ <= 10 or a_ >= 170:
                lines.append(ln)
    if not lines:
        return []
    n_ = len(lines)
    lookup = np.full((n_, n_), -1, dtype=int)
    for i in range(n_):
        for j in range(i+1, n_):
            d1 = np.hypot(lines[i][0] - lines[j][0], lines[i][1] - lines[j][1])
            d2 = np.hypot(lines[i][0] - lines[j][2], lines[i][1] - lines[j][3])
            d3 = np.hypot(lines[i][2] - lines[j][0], lines[i][3] - lines[j][1])
            d4 = np.hypot(lines[i][2] - lines[j][2], lines[i][3] - lines[j][3])
            mn = min(d1, d2, d3, d4)
            if mn < 30:
                c_ = np.where(lookup == j)
                if len(c_[0]) == 0:
                    cx_ = np.where(lookup == i)
                    if len(cx_[0]) == 0:
                        lookup[i, j] = j
                    else:
                        lookup[cx_[0][0], j] = j
                else:
                    lookup[c_[0][0], i] = i
    out = []
    for i in range(n_):
        c_ = np.where(lookup[i, :] != -1)
        if len(c_[0]) > 0:
            grp = [i] + list(c_[0])
            coords = []
            for g_ in grp:
                coords.append(lines[g_][0:4])
            coords = np.array(coords)
            allx = coords[:, [0, 2]].ravel()
            ally = coords[:, [1, 3]].ravel()
            mnx, mxx = allx.min(), allx.max()
            mny, mxy = ally.min(), ally.max()
            if axis == 2:
                out.append([mnx, mny, mnx, mxy, 1])
            else:
                out.append([mnx, mny, mxx, mny, 1])
        else:
            out.append([lines[i][0], lines[i][1],
                        lines[i][2], lines[i][3], 1])
    return out

def get_hog():
    return cv2.HOGDescriptor((100,100),(10,10),(5,5),(10,10),9,1,-1.,0,0.2,1,64,True)

def svm_predict(th2, rects, boxes):
    svm = cv2.ml.SVM_load("svm_data.dat")
    hog = get_hog()
    for (x1, y1, x2, y2) in rects:
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, th2.shape[1])
        y2 = min(y2, th2.shape[0])
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            continue
        reg = th2[y1:y2, x1:x2]
        reg = cv2.resize(reg, (100,100), interpolation=cv2.INTER_CUBIC)
        hd = hog.compute(reg)
        _, pred = svm.predict(np.array(hd, np.float32).reshape(-1, 3249))
        idx = int(pred[0][0]) + 3
        w_ = x2 - x1
        h_ = y2 - y1
        boxes.append([[x1, y1, w_, h_], idx])
    return boxes

def get_v_s_orientation(x, y, w, h, pairs):
    lines = []
    angle_ax = 0
    for i in range(len(pairs)):
        mx = (pairs[i,0] + pairs[i,2]) / 2
        my = (pairs[i,1] + pairs[i,3]) / 2
        if x < mx < x + w and y < my < y + h:
            if abs(pairs[i,0] - pairs[i,2]) > abs(pairs[i,1] - pairs[i,3]):
                angle_ax = 1
                val = my
            else:
                angle_ax = 0
                val = mx
            length = np.hypot(pairs[i,0] - pairs[i,2],
                              pairs[i,1] - pairs[i,3])
            lines.append([val, length])
    if len(lines) < 2:
        return 0
    arr = np.array(lines)
    arr = arr[arr[:,1].argsort()]
    if angle_ax == 0:
        if arr[1,0] > arr[0,0]:
            return 90
        else:
            return 270
    else:
        if arr[1,0] > arr[0,0]:
            return 180
        else:
            return 0

def get_diode_orientation(x, y, w, h, pairs):
    lines = []
    angle_ax = 0
    for i in range(len(pairs)):
        mx = (pairs[i,0] + pairs[i,2]) / 2
        my = (pairs[i,1] + pairs[i,3]) / 2
        if x < mx < x + w and y < my < y + h:
            if abs(pairs[i,0] - pairs[i,2]) > abs(pairs[i,1] - pairs[i,3]):
                angle_ax = 1
            else:
                angle_ax = 0
            lines.append([mx if angle_ax == 0 else my])
    if not lines:
        return 0
    arr = np.array(lines)
    if len(lines) == 1:
        angle_ax = 0
    else:
        angle_ax = 1
    if angle_ax == 0:
        if abs(arr[0,0] - x) > abs(arr[0,0] - (x + w)):
            return 270
        else:
            return 90
    else:
        if abs(arr[0,0] - y) > abs(arr[0,0] - (y + h)):
            return 0
        else:
            return 180

def output_file(wires, comp):
    lb = ['voltage','cap','ground','diode','res','ind']
    ab = ['V','C','G','D','R','L']
    off = [[0,16],[16,0],[0,0],[16,0],[16,16],[16,16]]
    ccount = np.zeros(6, dtype=int)
    fn = str(sys.argv[1])[:-4] + ".asc"
    fo = open(fn, "w", encoding='utf-8')
    fo.write("Version 4\nSHEET 1 880 680\n")
    for w_ in wires:
        fo.write(f"WIRE {int(w_[0])} {int(w_[1])} {int(w_[2])} {int(w_[3])}\n")
    for (bid, tid, n1, n2, (xx1, yy1), (xx2, yy2), ang) in comp:
        if tid == 2:
            fo.write(f"FLAG {xx1} {yy1} 0\n")
            continue
        if ang == 0:
            x_ = xx1 - off[tid][0]
            y_ = yy1 - off[tid][1]
        elif ang == 90:
            if tid == 0:
                x_ = xx1 + off[tid][1] * 6
                y_ = yy1
            elif tid == 3:
                x_ = xx1 + off[tid][0] * 4
                y_ = yy1 - off[tid][0]
            else:
                x_, y_ = xx1, yy1
        elif ang == 270:
            if tid == 0:
                x_ = xx1 - off[tid][1]
                y_ = yy1
            elif tid in [1, 3]:
                x_ = xx1
                y_ = yy1 + off[tid][0]
            elif tid in [4, 5]:
                x_ = xx1 - off[tid][0]
                y_ = yy1 + off[tid][0]
            else:
                x_, y_ = xx1, yy1
        elif ang == 180:
            if tid == 0:
                x_ = xx1 + off[tid][0]
                y_ = yy1 + off[tid][1] * 6
            elif tid == 3:
                x_ = xx1 + off[tid][0]
                y_ = yy1 + 64
            else:
                x_, y_ = xx1, yy1
        else:
            x_, y_ = xx1, yy1
        fo.write(f"SYMBOL {lb[tid]} {int(x_)} {int(y_)} R{ang}\n")
        fo.write(f"SYMATTR InstName {ab[tid]}{ccount[tid]}\n")
        ccount[tid] += 1
    fo.close()

def get_menubar():
    global process_stage
    bar = np.zeros((60,640,3), dtype=np.uint8)
    cv2.rectangle(bar, (0,0), (640,60), (239,239,239), -1)
    pc = (96,96,96)
    ac = (0,255,0)
    cv2.putText(bar, "Circuit Recognizer", (10,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, pc, 1)
    cv2.putText(bar, "Developer:", (10,35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, pc, 1)
    cv2.putText(bar, "Rohit Chakraborty", (82,35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,153,76), 1)
    cv2.putText(bar, "2025", (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, pc, 1)
    if process_stage == 0:
        cv2.rectangle(bar, (550,15), (620,45), (0,0,255), -1)
        cv2.putText(bar, "Segmentation", (230,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, ac, 1)
        cv2.putText(bar, "Classification", (330,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, pc, 1)
        cv2.putText(bar, "Result", (450,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, pc, 1)
        cv2.circle(bar, (270,40), 6, ac, -1)
        cv2.circle(bar, (370,40), 6, pc, -1)
        cv2.circle(bar, (470,40), 6, pc, -1)
        cv2.putText(bar, "next", (570,35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    elif process_stage == 1:
        cv2.rectangle(bar, (550,15), (620,45), (0,0,255), -1)
        cv2.putText(bar, "Segmentation", (230,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, pc, 1)
        cv2.putText(bar, "Classification", (330,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, ac, 1)
        cv2.putText(bar, "Result", (450,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, pc, 1)
        cv2.circle(bar, (270,40), 6, pc, -1)
        cv2.circle(bar, (370,40), 6, ac, -1)
        cv2.circle(bar, (470,40), 6, pc, -1)
        cv2.putText(bar, "next", (570,35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    else:
        cv2.rectangle(bar, (550,15), (620,45), (0,0,255), -1)
        cv2.putText(bar, "Segmentation", (230,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, pc, 1)
        cv2.putText(bar, "Classification", (330,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, pc, 1)
        cv2.putText(bar, "Result", (450,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        cv2.circle(bar, (270,40), 6, pc, -1)
        cv2.circle(bar, (370,40), 6, pc, -1)
        cv2.circle(bar, (470,40), 6, (0,255,0), -1)
        cv2.putText(bar, "finish", (570,35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    return bar

def draw_result_boxes(display, b):
    global process_stage
    if process_stage == 0:
        for (xx, yy, ww, hh), _ in b:
            cv2.rectangle(display, (xx, yy),
                          (xx + ww, yy + hh),
                          (0,255,0), 1)
    else:
        labels = ['v_source','capacitor','ground',
                  'diode','resistor','inductor']
        for ((x_, y_, w_, h_), idx_) in b:
            cv2.putText(display, labels[idx_], (x_ - 5, y_ - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (250,0,0), 1)
            cv2.rectangle(display, (x_, y_),
                          (x_ + w_, y_ + h_),
                          (0,255,0), 1)

def mouse_event_edit(event, x, y, flags, param):
    global edit_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        txt = ['v_source','capacitor','ground','diode','resistor','inductor']
        for i_ in range(len(txt)):
            if (20 < x < 120) and ((i_ * 40 + 20) < y < (i_ * 40 + 50)):
                boxes[param][1] = i_
                edit_flag = 1
                break

def mouse_event(event, x, y, flags, param):
    global process_stage, prev_stage, ix, iy, boxes, edit_flag, flagx
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if (560 < x < 630 and 500 < y < 530):
            prev_stage = process_stage
            process_stage += 1
    elif event == cv2.EVENT_RBUTTONDOWN:
        if process_stage == 0:
            ix, iy = x, y
        elif process_stage == 1:
            i = 0
            for (xx, yy, ww, hh), _ in boxes:
                if xx < x < xx + ww and yy < y < yy + hh:
                    edit_flag = i
                i += 1
            cv2.namedWindow("edit")
            cv2.moveWindow("edit", 960, 100)
            cv2.setMouseCallback("edit", mouse_event_edit, edit_flag)
            arr_ = ['v_source','capacitor','ground','diode','resistor','inductor']
            men = np.zeros((300,150,3), dtype=np.uint8)
            cv2.rectangle(men, (0,0), (150,300), (255,255,255), -1)
            for j_ in range(len(arr_)):
                cv2.rectangle(men, (20, j_ * 40 + 20),
                              (120, j_ * 40 + 50), (0,0,255), -1)
                cv2.putText(men, arr_[j_], (30, j_ * 40 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            while edit_flag != 1:
                cv2.imshow("edit", men)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyWindow("edit")
            edit_flag = 0
    elif event == cv2.EVENT_RBUTTONUP:
        if process_stage == 0:
            bw = cv2.imread("data/skel.pgm", 0)
            if bw is None:
                return
            y1, y2 = sorted([iy, y])
            x1, x2 = sorted([ix, x])
            y1 = max(y1, 0)
            y2 = min(y2, bw.shape[0])
            x1 = max(x1, 0)
            x2 = min(x2, bw.shape[1])
            if y2 <= y1 or x2 <= x1:
                return
            roi = bw[y1:y2, x1:x2]
            ends_ = skeleton_points(roi)
            for i_ in range(ends_[0].size):
                ends_[0][i_] += y1
                ends_[1][i_] += x1
            vp, hp = lines_between_ends(ends_)
            vb = box_between_ends(vp)
            hb = box_between_ends(hp)
            t_ = vb + hb
            if len(t_) == 1:
                boxes.append(t_[0])
            elif len(t_) > 1:
                th_img = cv2.imread("data/th.pgm", 0)
                local_rect = [[x1, y1, x2, y2]]
                local_box = []
                local_box = svm_predict(th_img, local_rect, local_box)
                if len(local_box) > 0:
                    boxes.append(local_box[0])
    elif event == cv2.EVENT_LBUTTONDOWN:
        if process_stage == 0:
            dlist = []
            i = 0
            for (xx, yy, ww, hh), _ in boxes:
                if xx < x < xx + ww and yy < y < yy + hh:
                    dlist.append(i)
                i += 1
            dlist.sort(reverse=True)
            for d_ in dlist:
                del boxes[d_]

if __name__ == "__main__":
    cv2.namedWindow("recognizer")
    cv2.moveWindow("recognizer", 200, 100)
    cv2.setMouseCallback("recognizer", mouse_event)

    src = cv2.imread(sys.argv[1])
    src = imutils.resize(src, width=640)
    org = src.copy()
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)
    th2 = th.copy()
    bw = thinning(th)
    cv2.imwrite("data/skel.pgm", bw)
    cv2.imwrite("data/th.pgm", th2)

    ends = skeleton_points(bw)
    vp, hp = lines_between_ends(ends)
    vb = box_between_ends(vp)
    hb = box_between_ends(hp)
    boxes = vb + hb

    for ((xx,yy,ww,hh), idd_) in boxes:
        th[yy:yy+hh, xx:xx+ww] = 0

    lines_ = lsd(th)
    for ln in lines_:
        x1, y1, x2, y2, w, *_ = ln
        ang_ = abs(np.rad2deg(np.arctan2(y1 - y2, x1 - x2)))
        if (75 < ang_ < 105) or ang_ > 160 or ang_ < 20:
            cv2.line(th, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,0), 6)

    kernel = np.ones((11,11), np.uint8)
    closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(closing.copy(),
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in cnts:
        c = c.astype(np.float32)
        if c.size < 6:
            continue
        if c.ndim == 2 and c.shape[1] == 2:
            c = c.reshape(-1,1,2)
        if c.shape[0] < 3:
            continue
        if c.ndim != 3 or c.shape[1] != 1 or c.shape[2] != 2:
            continue
        ar_ = cv2.contourArea(c)
        if ar_ < 80:
            continue
        x_, y_, w_, h_ = cv2.boundingRect(c)
        mx_ = max(w_, h_)
        x_ = int(((2*x_ + w_) - mx_)/2)
        y_ = int(((2*y_ + h_) - mx_)/2)
        rects.append([x_ - 10, y_ - 10,
                      x_ + mx_ + 10, y_ + mx_ + 10])

    boxes = svm_predict(th2, rects, boxes)

    while True:
        dsp = org.copy()
        if process_stage == 2 and prev_stage == 1:
            for ((xx,yy,ww,hh),ixx) in boxes:
                bw[yy:yy+hh, xx:xx+ww] = 0
                th2[yy:yy+hh, xx:xx+ww] = 0

            node_close = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)
            node_cnts_temp, _ = cv2.findContours(node_close.copy(),
                                                 cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)
            node_cnts = []
            node_ends = []
            if vp.size == 0 and hp.size == 0:
                pairs = np.empty((0,4), dtype=np.float32)
            elif vp.size == 0:
                pairs = hp
            elif hp.size == 0:
                pairs = vp
            else:
                pairs = np.vstack((vp, hp))

            i = 0
            for ccc_ in node_cnts_temp:
                if ccc_.size < 6:
                    continue
                ccc_ = ccc_.astype(np.float32)
                if ccc_.ndim == 2 and ccc_.shape[1] == 2:
                    ccc_ = ccc_.reshape(-1, 1, 2)
                if ccc_.shape[0] < 3:
                    continue
                if ccc_.ndim != 3 or ccc_.shape[1] != 1 or ccc_.shape[2] != 2:
                    continue
                area_n = cv2.contourArea(ccc_)
                if area_n <= 50:
                    continue
                node_cnts.append((i, ccc_))

                color = (randint(0,255), randint(0,255), randint(0,255))

                # final fallback check + try/except
                if ccc_.shape[0] < 3:
                    continue
                try:
                    cv2.drawContours(dsp, [ccc_], -1, color, 3)
                except cv2.error as e:
                    print("Skipping invalid contour. Reason:", e)
                    continue

                nm_ = np.zeros(th.shape, dtype=np.uint8)
                cv2.drawContours(nm_, [ccc_], -1, 255, 3)
                nth = thinning(nm_)
                e_ = skeleton_points(nth)
                for jj_ in range(e_[0].size):
                    xx_ = e_[1][jj_]
                    yy_ = e_[0][jj_]
                    node_ends.append([i, [xx_, yy_]])
                i += 1

            comp_ends = []
            i = 0
            for ((bx,by,bw_,bh_), idx_) in boxes:
                x1,y1 = int(bx + bw_/2), by
                x2,y2 = bx, int(by + bh_/2)
                x3,y3 = bx + bw_, int(by + bh_/2)
                x4,y4 = int(bx + bw_/2), by + bh_
                angle = 0
                minD = [99999, 99999]
                minI = [-1, -1, -1, -1]
                minE = [[-1,-1], [-1,-1]]
                k_ = 0
                for (ndid, (xe,ye)) in node_ends:
                    d1 = np.hypot(xe - x1, ye - y1)
                    d2 = np.hypot(xe - x2, ye - y2)
                    d3 = np.hypot(xe - x3, ye - y3)
                    d4 = np.hypot(xe - x4, ye - y4)
                    tmp_ = min(d1,d2,d3,d4)
                    if minD[0] > tmp_:
                        minD[1] = minD[0]
                        minI[1] = minI[0]
                        minE[1] = minE[0]
                        minD[0] = tmp_
                        minI[0] = ndid
                        minI[2] = k_
                        minE[0] = [xe, ye]
                    elif minD[1] > tmp_:
                        minD[1] = tmp_
                        minI[1] = ndid
                        minI[3] = k_
                        minE[1] = [xe, ye]
                    k_ += 1
                if idx_ == 2:  # ground
                    comp_ends.append([i, idx_, minI[0], minI[0],
                                      minE[0], minE[0], 0])
                    continue
                if abs(minE[0][0] - minE[1][0]) > abs(minE[0][1] - minE[1][1]):
                    angle = 270
                    if minE[1][0] < minE[0][0]:
                        tmpS = [minE[0][0], minE[0][1],
                                minI[0], minI[2]]
                        minE[0] = minE[1]
                        minE[1] = [tmpS[0], tmpS[1]]
                        minI[0] = minI[1]
                        minI[2] = minI[3]
                        minI[1] = tmpS[2]
                        minI[3] = tmpS[3]
                    if idx_ in [1,3]:
                        minE[0][0] -= 16
                        minE[1][0] = minE[0][0] + 64
                        minE[1][1] = minE[0][1]
                    elif idx_ in [0,4,5]:
                        minE[0][0] -= 20
                        minE[1][0] = minE[0][0] + 80
                        minE[1][1] = minE[0][1]
                else:
                    angle = 0
                    if minE[1][1] < minE[0][1]:
                        tmpS = [minE[0][0], minE[0][1],
                                minI[0], minI[2]]
                        minE[0] = minE[1]
                        minE[1] = [tmpS[0], tmpS[1]]
                        minI[0] = minI[1]
                        minI[2] = minI[3]
                        minI[1] = tmpS[2]
                        minI[3] = tmpS[3]
                    if idx_ in [1,3]:
                        minE[1][0] = minE[0][0]
                        minE[0][1] -= 16
                        minE[1][1] = minE[0][1] + 64
                    elif idx_ in [0,4,5]:
                        minE[1][0] = minE[0][0]
                        minE[0][1] -= 20
                        minE[1][1] = minE[0][1] + 80
                if idx_ == 0:
                    angle = get_v_s_orientation(bx, by, bw_, bh_, pairs)
                elif idx_ == 3:
                    angle = get_diode_orientation(bx, by, bw_, bh_, pairs)
                comp_ends.append([i, idx_, minI[0], minI[1],
                                  minE[0], minE[1], angle])
                i += 1

            tmpn = []
            for (bid, tid, n1, n2, (xx1, yy1), (xx2, yy2), _) in comp_ends:
                tmpn.append([n1, [xx1, yy1]])
                tmpn.append([n2, [xx2, yy2]])
            node_ends = deepcopy(tmpn)

            wires = []
            refs = []
            for (nid, c_) in node_cnts:
                if c_.size < 6:
                    continue
                if c_.shape[0] < 3:
                    continue
                mask_ = np.zeros(th.shape, dtype=np.uint8)
                cv2.drawContours(mask_, [c_], -1, 255, 3)
                node_ls = lsd(mask_)
                tv = remove_parallel_lines(node_ls, 2)
                thl = remove_parallel_lines(node_ls, 1)
                all_ = tv + thl
                if len(all_) == 0:
                    continue
                big_ = 0
                j_ = -1
                for (ax1, ay1, ax2, ay2, _), idd_ in zip(all_, range(len(all_))):
                    dd_ = np.hypot(ax2 - ax1, ay2 - ay1)
                    if dd_ > big_:
                        big_ = dd_
                        j_ = idd_
                if j_ < 0:
                    continue
                ref = np.array([[all_[j_][0], all_[j_][1]],
                                [all_[j_][2], all_[j_][3]]], dtype=np.float32)
                n_ = []
                for (jj, (xe, ye)) in node_ends:
                    if jj == nid:
                        n_.append([xe, ye])
                if len(n_) > 0:
                    n_ = np.array(n_)
                    if abs(ref[0][0] - ref[1][0]) > abs(ref[0][1] - ref[1][1]):
                        n_ = n_[n_[:,0].argsort()]
                        ref = ref[ref[:,0].argsort()]
                        ref[0][0] = n_[0][0]
                        ref[1][0] = n_[-1][0]
                        d1_ = abs(ref[0][1] - n_[0][1])
                        d2_ = abs(ref[1][1] - n_[-1][1])
                        if d1_ < 10:
                            ref[0][1] = n_[0][1]
                            ref[1][1] = n_[0][1]
                        elif d2_ < 10:
                            ref[0][1] = n_[-1][1]
                            ref[1][1] = n_[-1][1]
                        else:
                            ref[0][1] = ref[1][1]
                    else:
                        n_ = n_[n_[:,1].argsort()]
                        ref = ref[ref[:,1].argsort()]
                        ref[0][1] = n_[0][1]
                        ref[1][1] = n_[-1][1]
                        d1_ = abs(ref[0][0] - n_[0][0])
                        d2_ = abs(ref[1][0] - n_[-1][0])
                        if d1_ < 10:
                            ref[0][0] = n_[0][0]
                            ref[1][0] = n_[0][0]
                        elif d2_ < 10:
                            ref[0][0] = n_[-1][0]
                            ref[1][0] = n_[-1][0]
                        else:
                            ref[0][0] = ref[1][0]
                refs.append([ref[0][0], ref[0][1],
                             ref[1][0], ref[1][1], 1])

            for (bid, tid, n1, n2, (xx1, yy1), (xx2, yy2), _) in comp_ends:
                if abs(refs[n1][0] - refs[n1][2]) > abs(refs[n1][1] - refs[n1][3]):
                    xx_ = xx1
                    yy_ = refs[n1][1]
                else:
                    xx_ = refs[n1][0]
                    yy_ = yy1
                wires.append([xx1, yy1, xx_, yy_, 1])

                if abs(refs[n2][0] - refs[n2][2]) > abs(refs[n2][1] - refs[n2][3]):
                    xx_ = xx2
                    yy_ = refs[n2][1]
                else:
                    xx_ = refs[n2][0]
                    yy_ = yy2
                wires.append([xx2, yy2, xx_, yy_, 1])

            wires.extend(refs)
            output_file(wires, comp_ends)
            flagx = 1

        menubar = get_menubar()
        draw_result_boxes(dsp, boxes)
        stack_ = np.vstack((dsp, menubar))
        cv2.imshow("recognizer", stack_)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or flagx == 1:
            break

    while process_stage < 3:
        cv2.imshow("recognizer", stack_)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
