#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import copy
import cv2



# definiton refers to navfun_ros.cpp
# map index <<==>> coordinates
#         nx(x size)     
#                ____________  mx
#               | 0  1  2  3 
#               | 4  5  6  7
# ny(y size)    | > > > > > >
#           my  | > > > > > >   index increase
# index = my * nx + mx
# cooridnate index collide with the matrix index
POT_HIGH =1e10 
COST_OBS = 254
COST_OBS_ROS = 253
PRIORITYBUFSIZE = 10000
INVSQRT2 = 0.707107781
COST_NEUTRAL = 50


cell_Count = 0


class NavFun:
    
    def __init__(self, x_size, y_size, costarr):
        self.x_size = x_size
        self.y_size = y_size
        self.potarr = np.ones((x_size*y_size))*POT_HIGH 
        self.costarr = costarr.flatten()
        self.curT = COST_OBS
        self.curP = np.zeros(PRIORITYBUFSIZE, dtype=np.int32)
        self.curPe = 0
        self.nextP = np.zeros(PRIORITYBUFSIZE, dtype=np.int32)
        self.nextPe = 0
        self.overP = np.zeros(PRIORITYBUFSIZE, dtype=np.int32)
        self.overPe = 0
        self.pending = np.full(x_size*y_size, False) 
        self.goal = np.array([159,209]) #np.array([40,40]) # np.array([159,209])
        self.start = np.array([224,207])#np.array([60,20]) #np.array([224,207])
        self.npath = 0
        self.npathbuff = 0
        self.pathStep = 0.5
        self.gradx = np.zeros(x_size*y_size)
        self.grady = np.zeros(x_size*y_size)
        self.priInc = 2*COST_NEUTRAL
        self.found = False
        
    
    def drawCostmap(self):

        plot_map_arr = self.costarr  #-1*(self.costarr - 255)
        plt.figure(1)
        plt.title("costarr and path")

        plt.imshow(plot_map_arr.reshape(self.y_size, self.x_size), cmap='gray')
        plt.plot(self.start[0], self.start[1],'ro')
        plt.plot(self.goal[0], self.goal[1],'g*')

        print("pathx: ", self.pathx[:self.npath])
        print("pathy: ", self.pathy[:self.npath])
        if self.found:
            plt.plot(self.pathx[:self.npath], self.pathy[:self.npath],'r-')

        plt.grid()

        plt.figure(2)
        print("potarr: ", self.potarr)
        plt.imshow(self.potarr.reshape(self.y_size, self.x_size), cmap='gray')
        plt.title("potarr")
        plt.show()


    def setupNavFn(self):
        nx = self.x_size
        ny = self.y_size
        for i in range(nx):
            self.costarr[i] = COST_OBS
            self.costarr[(self.y_size-1)*self.x_size+i] = COST_OBS
        for i in range(ny): 
            self.costarr[i*self.x_size+0] = COST_OBS
            self.costarr[i*self.x_size+self.x_size-1] = COST_OBS
        self.initCost(self.goal, 0)


    def initCost(self, goal, v):
        k = goal[1]*self.x_size+goal[0]
        self.potarr[k] = v
        # k = pos[0] + pos[1]* self.x_size
        print("k: ", k)
        self.push_cur(k + 1)
        self.push_cur(k - 1)
        self.push_cur(k - self.x_size)
        self.push_cur(k + self.x_size)
    

    def push_cur(self, n):
        if n >=0 and n < self.x_size* self.y_size and not self.pending[n] and\
                self.costarr[n] < COST_OBS and self.curPe < PRIORITYBUFSIZE:
            self.curP[self.curPe] = n
            self.pending[n] = True
            self.curPe +=1

    def push_next(self, n):
        if n >=0 and n < self.x_size* self.y_size and not self.pending[n] and\
                self.costarr[n] < COST_OBS and self.nextPe < PRIORITYBUFSIZE:
            self.nextP[self.nextPe] = n
            self.pending[n] = True
            self.nextPe +=1
    
    def push_over(self, n):
        if n >=0 and n < self.x_size* self.y_size and not self.pending[n] and\
                self.costarr[n] < COST_OBS and self.overPe < PRIORITYBUFSIZE:
            self.overP[self.overPe] = n
            self.pending[n] = True
            self.overPe +=1

    
    def calcNavFnDijkstra(self, atStart):
        self.setupNavFn()
        self.propNavFnDijkstra(max(self.x_size*self.y_size/20, self.x_size+self.y_size), atStart)
        len_size = self.calcPath(np.int32(self.x_size*self.y_size/2))

        if len_size > 0:
            print("NavFun Path found")
            self.found = True
            return True
        else:
            print("NavFun Path not found")
            return False


    
    def propNavFnDijkstra(self, cycles, atStart):

        global cell_Count
        
        nwv = 0    # max priority block size
        nc = 0     # the number of cells put into priority block
        cycle = 0  # which cycle we're on
        
        while cycle < cycles:

            if self.curPe == 0 and self.nextPe == 0:
                break

            nc += self.curPe

            if self.curPe > nwv:
                nwv = self.curPe

            # reset pending flags on current priority block
            i = self.curPe

            count = 0 
            while i > 0:
                self.pending[self.curP[count]] = False
                count += 1
                i -= 1

            i = self.curPe
            dog1 = self.potarr
            # print(dog1)
            count = 0
            while i > 0:
                self.updateCell(self.curP[count])
                count += 1
                i -= 1
                cell_Count += 1

            self.curPe = self.nextPe
            self.nextPe = 0
            swap = copy.copy(self.curP)
            self.curP = copy.copy(self.nextP)
            self.nextP = copy.copy(swap)

            if self.curPe == 0:
                self.curT += self.priInc
                self.curPe = self.overPe
                self.overPe = 0
                swap = copy.copy(self.curP)
                self.curP = copy.copy(self.overP)
                self.overP = copy.copy(swap)


            
            if (atStart):
                if (self.potarr[self.start[0]+ self.start[1]*self.x_size] < POT_HIGH):
                    break

            cycle += 1
        if cycle < cycles:
            return True
        else:
            return False


    def updateCell(self, k):
        global cell_Count
        dog = self.potarr
        l = self.potarr[k-1]
        r = self.potarr[k+1]
        u = self.potarr[k-self.x_size]
        d = self.potarr[k+ self.x_size]

        if l < r:
            tc = l
        else:
            tc = r

        if u < d:
            ta = u
        else:
            ta = d
        if  self.costarr[k] < COST_OBS:
            hf = np.float32(self.costarr[k])
            dc = tc - ta
            if dc < 0:
                dc = - dc
                ta = tc
            if dc >= hf:
                pot = ta + hf
            else:
                dd = dc / hf
                v = -0.2301*dd*dd + 0.5307*dd + 0.7040
                pot = ta + hf*v

            if pot < self.potarr[k]:
                le = INVSQRT2 * np.float32(self.costarr[k-1])
                re = INVSQRT2 * np.float32(self.costarr[k+1])
                ue = INVSQRT2 * np.float32(self.costarr[k-self.x_size])
                de = INVSQRT2 * np.float32(self.costarr[k+self.x_size])
    
                self.potarr[k] = pot
    
                # n = sx + sy * self.x_size

                if pot < self.curT:
                    if l > pot + le:
                        self.push_next(k-1)
                    if r > pot + re:  # r -> potarr element
                        self.push_next(k+1)
                    if u > pot + ue:
                        self.push_next(k - self.x_size)
                    if d > pot + de:
                        self.push_next(k + self.x_size)
                else:
                    if l > pot + le:
                        self.push_over(k-1)
                    if r > pot + re:  # r -> potarr element
                        self.push_over(k+1)
                    if u > pot + ue:
                        self.push_over(k - self.x_size)
                    if d > pot + de:
                        self.push_over(k + self.x_size)


                                    
    def calcPath(self, n):
        if self.npathbuff < n:
            self.npathbuff = n
        st = copy.copy(self.start)
        stc = st[0] + st[1]* self.x_size
        print("stc 0: ", stc)
        dx = 0
        dy = 0
        self.npath = 0
        self.pathx = np.zeros(n)
        self.pathy = np.zeros(n)


        for i in range(n):
            nearest_point = max(0, min(self.x_size*self.y_size-1, stc + np.int32(np.round(dx))+np.int32(self.x_size*np.round(dy)))) 

            x0, y0 = self.index2Coordinate(nearest_point)

            # print("nearest_point: ", nearest_point)

            if (self.potarr[nearest_point] < COST_NEUTRAL):
                self.pathx[self.npath] = np.float32(self.goal[0])
                self.pathy[self.npath] = np.float32(self.goal[1])

                self.npath += 1
                return self.npath
            if stc < self.x_size or stc > (self.y_size-1)*self.x_size:
                raise ValueError("PathCalc out of the bounds")
                return 0
            
            self.pathx[self.npath] = stc % self.x_size + dx
            self.pathy[self.npath] = np.int32(stc/ self.x_size) + dy
            self.npath += 1
            
            oscillation_detected = False
            if (self.npath > 2 and self.pathx[self.npath-1] == self.pathx[self.npath-3] and self.pathy[self.npath-1] == self.pathy[self.npath-3]):
                print("PathCalc oscillation detected, attempted to fix it")
                oscillation_detected = True

            stcnx = stc + self.x_size
            stcpx = stc - self.x_size

            # check for potentials at 8 positions near cell

            stcx, stcy = self.index2Coordinate(stc)

            if (self.potarr[stc] >= POT_HIGH) or self.potarr[stc+1] >= POT_HIGH or self.potarr[stc-1] >= POT_HIGH or self.potarr[stcnx] >= POT_HIGH or self.potarr[stcnx+1] >= POT_HIGH or self.potarr[stcnx-1] >=POT_HIGH or \
                    self.potarr[stcpx] >= POT_HIGH or self.potarr[stcpx+1]>= POT_HIGH or self.potarr[stcpx-1]>= POT_HIGH or oscillation_detected:
                
                minc = stc;
                minp = self.potarr[stc]

                stt = stcpx - 1
                
                # print("pot1: ", self.potarr[stcx-1, stcy-1])
                if self.potarr[stt] < minp:
                    minp = self.potarr[stt]
                    minc = stt
                stt += 1

                stx, sty = self.index2Coordinate(stt)


                # print("pot2: ", self.potarr[stx, sty])

                if self.potarr[stt] < minp:
                    minp = self.potarr[stt]
                    minc = stt
                stt += 1
                stx, sty = self.index2Coordinate(stt)


                # print("pot3: ", self.potarr[stx, sty])
                if self.potarr[stt] < minp:
                    minp = self.potarr[stt]
                    minc = stt
                stt = stc - 1
                
                stx, sty = self.index2Coordinate(stt)

                # print("pot4: ", self.potarr[stx, sty])
                if self.potarr[stt] < minp:
                    minp = self.potarr[stt]
                    minc = stt
                stt = stc + 1
                stx, sty = self.index2Coordinate(stt)

                # print("pot5: ", self.potarr[stx, sty])
                if self.potarr[stt] < minp:
                    minp = self.potarr[stt]
                    minc = stt
                stt = stcnx - 1
                stx, sty = self.index2Coordinate(stt)

                # print("pot6: ", self.potarr[stx, sty])
                if self.potarr[stt] < minp:
                    minp = self.potarr[stt]
                    minc = stt
                stt += 1
                stx, sty = self.index2Coordinate(stt)
                

                # print("pot7: ", self.potarr[stx, sty])
                if self.potarr[stt] < minp:
                    minp = self.potarr[stt]
                    minc = stt
                stt += 1
                stx, sty = self.index2Coordinate(stt)


                # print("pot8: ", self.potarr[stx, sty])
                if self.potarr[stt] < minp:
                    minp = self.potarr[stt]
                    minc = stt
                stc = minc
                
                stcx, stcy = self.index2Coordinate(stc)
                dx = 0
                dy = 0
                
                if (self.potarr[stc] >= POT_HIGH):
                    
                    return 0
            else:
                self.gradCell(stc)
                self.gradCell(stc+1)
                self.gradCell(stcnx)
                self.gradCell(stcnx+1)

                x1 = (1.0 - dx)*self.gradx[stc] + dx*self.gradx[stc+1]
                x2 = (1.0 - dx)*self.gradx[stcnx] + dx*self.gradx[stcnx+1]
                x = (1.0 - dy) * x1 + dy * x2
                
                y1 = (1.0 - dx)*self.grady[stc] + dx*self.grady[stc+1]
                y2 = (1.0 - dx)*self.grady[stcnx] + dx*self.grady[stcnx+1]
                y = (1.0 - dy) * y1 + dy * y2

                if x==0.0 and y==0.0:
                    print("PathCalc Zero gradient")
                    return 0

                ss = self.pathStep / np.sqrt(x**2 + y**2)
                dx += x * ss
                dy += y * ss

                if (dx > 1.0):
                    stc += 1
                    dx -= 1.0
                if dx < -1.0:
                    stc -= 1
                    dx += 1.0
                if (dy > 1.0):
                    stc += self.x_size
                    dy -= 1.0
                if dy < -1.0:
                    stc -= self.x_size
                    dy += 1.0
    
        return 0


    def index2Coordinate(self, n):

        sy = np.int32(n / self.x_size) 
        sx = n - sy * self.x_size
        return sx, sy


    def gradCell(self, n):
        if self.gradx[n] + self.grady[n] >0.0:
            return 1.0
        if n < self.x_size or n > (self.y_size - 1)* self.x_size:
            return 0.0
        x1, y1 = self.index2Coordinate(n)
        cv = self.potarr[n]
        dx = 0.0
        dy = 0.0
        
        # check for in an obstacle
        if cv >= POT_HIGH:
            if self.potarr[n-1]  < POT_HIGH:
                dx = -COST_OBS
            elif self.potarr[n+1] < POT_HIGH:
                dx = COST_OBS

            if self.potarr[n-self.x_size] < POT_HIGH:
                dy = -COST_OBS
            elif self.potarr[n+self.x_size] < POT_HIGH:
                dy = COST_OBS
        else:                                    # not in an obstacle
            if self.potarr[n-1] < POT_HIGH:
                dx += self.potarr[n-1] - cv
            if self.potarr[n+1] < POT_HIGH:
                dx += cv - self.potarr[n+1]
            if self.potarr[n-self.x_size] < POT_HIGH:
                dy += self.potarr[n-self.x_size] - cv
            if self.potarr[n+self.x_size] < POT_HIGH:
                dy += cv - self.potarr[n+self.x_size]

        # normalize
        norm = np.sqrt(dx**2+ dy**2)
        if norm > 0:
            norm = 1.0 / norm
            self.gradx[n] = norm * dx
            self.grady[n] = norm * dy

        return norm



mapdata = np.loadtxt("./costmap_nav.txt")
# mapdata = np.array(cv2.imread("./map.png", cv2.IMREAD_GRAYSCALE))
x_,y_ = mapdata.shape

print("x_size: ", y_)
print("y_size: ", x_)
"""
for i in range(x_):
    for j in range(y_):
        if mapdata[i, j] ==255:
            mapdata[i,j] = 254
"""
navfun = NavFun(y_, x_, mapdata) # input x_size, y_size , map

navfun.calcNavFnDijkstra(True)

navfun.drawCostmap()





