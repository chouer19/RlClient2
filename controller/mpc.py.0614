
import sys
sys.path.append("../libs/CubicSpline/")
import numpy as np
import cvxpy
import matplotlib.pyplot as plt
import cubic_spline_planner

sys.path.append("../libs")
from vehicle import *
from road import *

import time
import pygame
from pygame.locals import *

#dependent function
def get_nparray_from_matrix(x):
    return np.array(x).flatten()

BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)
DARKPINK = (255,20,147)
DARKRED = (138,0,0)
PURPLE = (160,32,240)
YELLOW = (255,255,0)
GREEN = (00,255,0)
BLUE = (0,0,255)
LIGHTBLUE = (176,226,255)
ORANGE = (139,69,0)

WIDTH = 1200
HEIGHT = 1000
PIXEL_DENSITY = 2.8

FULL_OFFSET_X = 25
FULL_OFFSET_Y = 25

CAR_WIDTH = 2.4
CAR_LENGTH = 5.5

class MPC:
    def __init__(self,filename = None):
        self.NX = 4  # x = x, y, v, yaw
        self.NU = 2  # a = [accel, steer]
        self.T = 5  # horizon length
        self.HorizonLength = 5
        #self.HorizonLength = 10
        self.R = np.diag([0.01, 0.01])  # input cost matrix
        self.Rd = np.diag([0.01, 1.0])  # input difference cost matrix
        self.Q = np.diag([1.0, 1.0, 0.5, 0.5])  # state cost matrix
        self.Qf = self.Q  # state final matrix
        self.GOAL_DIS = 1.5  # goal distance
        self.GOAL_DIS = 5  # goal distance
        self.STOP_SPEED = 0.5 / 3.6  # stop speed
        self.MAX_TIME = 500.0  # max simulation time

        # iterative paramter
        self.MAX_ITER = 5  # Max iteration
        self.MAX_ITER = 8  # Max iteration
        self.DU_TH = 0.1  # iteration finish param
        self.N_IND_SEARCH = 10  # Search index number
    
        self.show_animation = True

        self.DT = 0.2 # time tick (s)
        self.DT = 0.1 # time tick (s)

        self.vehicle = Vehicle()
        self.road = Road()
        if filename is not None:
            self.road = Road(filename)
        self.state = State()

        self.step = 0
        self.reward = 0
        self.eDis = 0
        self.eYaw = 0

        self.oa = [0.0] * self.HorizonLength
        self.odelta = [0.0] * self.HorizonLength

        #init pygame
        self.fullscreen = True
        pygame.init()
        self._display = pygame.display.set_mode( (WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._line = pygame.display.set_mode( (WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.flip()
        self.map_points = self._generate_pointlist()
        self.trajectory = []

    def render(self):
        self._line.fill(BLACK)
        self._display.fill(BLACK)
        self.trajectory.append(( int(self.state.x * PIXEL_DENSITY + FULL_OFFSET_X), int(self.state.y * PIXEL_DENSITY + FULL_OFFSET_Y)  ) )
        if len(self.trajectory) > 1000:
            self.trajectory = self.trajectory[5:-1]
        if self.fullscreen:
            pygame.draw.lines(self._line,GREEN,False,self.map_points, 2)
            pygame.draw.circle(self._line, (255,255,255), [int(self.state.x * PIXEL_DENSITY + FULL_OFFSET_X), int(self.state.y * PIXEL_DENSITY + FULL_OFFSET_Y)], 3)
            pygame.draw.lines(self._line, RED, False, \
                                        [ ( int(self.state.x * PIXEL_DENSITY + FULL_OFFSET_X), int(self.state.y * PIXEL_DENSITY + FULL_OFFSET_Y) ),\
                                         ( int(self.state.x * PIXEL_DENSITY + math.cos(self.state.yaw)* 20 + FULL_OFFSET_X), int(self.state.y * PIXEL_DENSITY + math.sin(self.state.yaw)*20 + FULL_OFFSET_Y) )] ,3)
            if len(self.trajectory) > 10:
                pygame.draw.lines(self._line, ORANGE,False,self.trajectory, 2)
            pass
        else:
            
            pass
        pass
        for event in pygame.event.get():
            pass
        keys = pygame.key.get_pressed()
        #if keys[K_SPACE] or keys[K_UP] or keys[K_DOWN]:
        #    self.fullscreen = not self.fullscreen
        #if keys[K_RETURN]:
        #    print("pause 5s")
        #    time.sleep(5)
        pygame.display.flip()
        pygame.display.update()

    def _generate_pointlist(self):
        map_points = []
        for (cx,cy) in zip(self.road.cx,self.road.cy):
            map_points.append( (int(cx * PIXEL_DENSITY + FULL_OFFSET_X),int(cy * PIXEL_DENSITY + FULL_OFFSET_Y)) )
        return map_points

    def get_linear_model_matrix(self,v, phi, delta):
        A = np.matrix(np.zeros((self.NX, self.NX)))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.DT * math.cos(phi)
        A[0, 3] = - self.DT * v * math.sin(phi)
        A[1, 2] = self.DT * math.sin(phi)
        A[1, 3] = self.DT * v * math.cos(phi)
        A[3, 2] = self.DT * math.tan(delta)
    
        B = np.matrix(np.zeros((self.NX, self.NU)))
        B[2, 0] = self.DT
        B[3, 1] = self.DT * v / (self.vehicle.WB * math.cos(delta) ** 2)
    
        C = np.matrix(np.zeros((self.NX, 1)))
        C[0, 0] = self.DT * v * math.sin(phi) * phi
        C[1, 0] = - self.DT * v * math.cos(phi) * phi
        C[3, 0] = v * delta / (self.vehicle.WB * math.cos(delta) ** 2)
    
        return A, B, C
    
    
    def plot_car(self, x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):
        outline = np.matrix([[-self.vehicle.BACKTOWHEEL, (self.vehicle.LENGTH - self.vehicle.BACKTOWHEEL), (self.vehicle.LENGTH - self.vehicle.BACKTOWHEEL), -self.vehicle.BACKTOWHEEL, -self.vehicle.BACKTOWHEEL],
                             [self.vehicle.WIDTH / 2, self.vehicle.WIDTH / 2, - self.vehicle.WIDTH / 2, -self.vehicle.WIDTH / 2, self.vehicle.WIDTH / 2]])
    
        fr_wheel = np.matrix([[self.vehicle.WHEEL_LEN, -self.vehicle.WHEEL_LEN, -self.vehicle.WHEEL_LEN, self.vehicle.WHEEL_LEN, self.vehicle.WHEEL_LEN],
                              [-self.vehicle.WHEEL_WIDTH - self.vehicle.TREAD, -self.vehicle.WHEEL_WIDTH - self.vehicle.TREAD, self.vehicle.WHEEL_WIDTH - self.vehicle.TREAD, self.vehicle.WHEEL_WIDTH - self.vehicle.TREAD, -self.vehicle.WHEEL_WIDTH - self.vehicle.TREAD]])
    
        rr_wheel = np.copy(fr_wheel)
    
        fl_wheel = np.copy(fr_wheel)
        fl_wheel[1, :] *= -1
        rl_wheel = np.copy(rr_wheel)
        rl_wheel[1, :] *= -1
    
        Rot1 = np.matrix([[math.cos(yaw), math.sin(yaw)],
                          [-math.sin(yaw), math.cos(yaw)]])
        Rot2 = np.matrix([[math.cos(steer), math.sin(steer)],
                          [-math.sin(steer), math.cos(steer)]])
    
        fr_wheel = (fr_wheel.T * Rot2).T
        fl_wheel = (fl_wheel.T * Rot2).T
        fr_wheel[0, :] += self.vehicle.WB
        fl_wheel[0, :] += self.vehicle.WB
    
        fr_wheel = (fr_wheel.T * Rot1).T
        fl_wheel = (fl_wheel.T * Rot1).T
    
        outline = (outline.T * Rot1).T
        rr_wheel = (rr_wheel.T * Rot1).T
        rl_wheel = (rl_wheel.T * Rot1).T
    
        outline[0, :] += x
        outline[1, :] += y
        fr_wheel[0, :] += x
        fr_wheel[1, :] += y
        rr_wheel[0, :] += x
        rr_wheel[1, :] += y
        fl_wheel[0, :] += x
        fl_wheel[1, :] += y
        rl_wheel[0, :] += x
        rl_wheel[1, :] += y
    
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), truckcolor)
        plt.plot(np.array(fr_wheel[0, :]).flatten(),
                 np.array(fr_wheel[1, :]).flatten(), truckcolor)
        plt.plot(np.array(rr_wheel[0, :]).flatten(),
                 np.array(rr_wheel[1, :]).flatten(), truckcolor)
        plt.plot(np.array(fl_wheel[0, :]).flatten(),
                 np.array(fl_wheel[1, :]).flatten(), truckcolor)
        plt.plot(np.array(rl_wheel[0, :]).flatten(),
                 np.array(rl_wheel[1, :]).flatten(), truckcolor)
        plt.plot(x, y, "*")
    
    
    def update_state(self,state, a, delta):
        # input check
        if delta >= self.vehicle.MAX_STEER:
            delta = self.vehicle.MAX_STEER
        elif delta <= -self.vehicle.MAX_STEER:
            delta = -self.vehicle.MAX_STEER
    
        state.x = state.x + state.v * math.cos(state.yaw) * self.DT
        state.y = state.y + state.v * math.sin(state.yaw) * self.DT
        state.yaw = state.yaw + state.v / self.vehicle.WB * math.tan(delta) * self.DT
        #state.yaw = pi_2_pi( state.yaw + state.v / self.vehicle.WB * math.tan(delta) * self.DT )
        state.v = state.v + a * self.DT
    
        if state. v > self.vehicle.MAX_SPEED:
            state.v = self.vehicle.MAX_SPEED
        elif state. v < self.vehicle.MIN_SPEED:
            state.v = self.vehicle.MIN_SPEED
    
        return state
    
    
    def iterative_linear_mpc_control(self, xref, x0, od):
        """
        MPC contorl with updating operational point iteraitvely
        """
    
        dref = np.zeros((1, self.HorizonLength + 1))

        if od is None:
            od = [0.0] * self.HorizonLength
    
        for i in range(self.MAX_ITER):
            xbar = self.predict_motion(x0, od, xref)
            poa, pod = oa[:], od[:]
            oa, od, ox, oy, oyaw, ov = self.linear_mpc_control(xref, xbar, x0, dref)
            du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
            if du <= self.DU_TH:
                break
        else:
            print("Iterative is max iter")
    
        return oa, od, ox, oy, oyaw, ov
    
    def linear_mpc_control(self,xref, xbar, x0, dref):
        """
        linear mpc control
    
        xref: reference point
        xbar: operational point
        x0: initial state
        dref: reference steer angle
        """
    
        x = cvxpy.Variable(self.NX, self.HorizonLength + 1)
        u = cvxpy.Variable(self.NU, self.HorizonLength)
    
        cost = 0.0
        constraints = []
    
        for t in range(self.HorizonLength):
            cost += cvxpy.quad_form(u[:, t], self.R)
    
            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], self.Q)
    
            A, B, C = self.get_linear_model_matrix(
                xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A * x[:, t] + B * u[:, t] + C]
    
            if t < (self.T - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], self.Rd)
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t])
                                < self.vehicle.MAX_DSTEER * self.DT]
    
        cost += cvxpy.quad_form(xref[:, self.HorizonLength] - x[:, self.HorizonLength], self.Qf)
    
        constraints += [x[:, 0] == x0]
        constraints += [x[2, :] <= self.vehicle.MAX_SPEED]
        constraints += [x[2, :] >= self.vehicle.MIN_SPEED]
        constraints += [cvxpy.abs(u[0, :]) < self.vehicle.MAX_ACCEL]
        constraints += [cvxpy.abs(u[1, :]) < self.vehicle.MAX_STEER]
    
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(verbose=False)
    
        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox = get_nparray_from_matrix(x.value[0, :])
            oy = get_nparray_from_matrix(x.value[1, :])
            ov = get_nparray_from_matrix(x.value[2, :])
            oyaw = get_nparray_from_matrix(x.value[3, :])
            oa = get_nparray_from_matrix(u.value[0, :])
            odelta = get_nparray_from_matrix(u.value[1, :])
    
        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None
    
        return oa, odelta, ox, oy, oyaw, ov
    
    
    def calc_ref_trajectory(self):
        xref = np.zeros((self.NX, self.HorizonLength + 1))
        ncourse = len(self.road.cx)
    
        ind, dis = self.road.findNearest(self.state.x, self.state.y)
    
        xref[0, 0] = self.road.cx[ind]
        xref[1, 0] = self.road.cy[ind]
        xref[2, 0] = self.road.sp[ind]
        xref[3, 0] = self.road.cyaw[ind]
    
        travel = 0.0
    
        #for i in range(T + 1):
        for i in range(self.HorizonLength + 1):
            travel += abs(self.state.v) * self.DT
            dind = int(round(travel / self.road.dl))
    
            xref[0, i] =   self.road.cx[(ind + dind)%ncourse]
            xref[1, i] =   self.road.cy[(ind + dind)%ncourse]
            xref[2, i] =   self.road.sp[(ind + dind)%ncourse]
            xref[3, i] = self.road.cyaw[(ind + dind)%ncourse]
    
        return xref, ind, dis
    
    #dependent function
    def predict_motion(self, x0, od, xref):
        xbar = xref * 0.0
        for i in range(len(x0)):
            xbar[i, 0] = x0[i]
    
        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (di, i) in zip(od, range(1, self.HorizonLength + 1)):
            state = self.update_state(state, self.state.acc, di)
            xbar[0, i] = state.x
            xbar[1, i] = state.y
            xbar[2, i] = state.v
            xbar[3, i] = state.yaw
    
        return xbar

    def log(self):
        pass

    def control_steer(self):
        #xref, ind, dis = self.road.calc_ref_trajectory(self.state, self.HorizonLength)
        xref, ind, dis = self.calc_ref_trajectory()
        x0 = [self.state.x, self.state.y, self.state.v, self.state.yaw]  # current state
        self.oa, self.odelta, ox, oy, oyaw, ov = self.iterative_linear_mpc_control( xref, x0, self.odelta )

        self.render()
        return self.odelta[0]

#dependent function
def main():
    print(__file__ + " start!!")

    #mpc = MPC()
    #mpc.road = Road(filename = '../log/all.road')
    mpc = MPC(filename = '../log/all.road')
    #mpc.road.test()
    def do_simulation2(mpc):
        #get state
        #state = State(x=mpc.cx[0], y=mpc.cy[0], yaw=mpc.cyaw[0], v=0.0)
        mpc.state = State(x=mpc.road.cx[0], y=mpc.road.cy[0] + 2, yaw=mpc.road.cyaw[0], v=0.0)
        times = 0.0
        odelta, oa = None, None
        ai,di = 0,0 
        # get new state
        mpc.state = mpc.update_state(mpc.state, ai, di)

        x = [mpc.state.x]
        y = [mpc.state.y]
        yaw = [mpc.state.yaw]
        v = [mpc.state.v]
        t = [0.0]
        d = [0.0]
        a = [0.0]


        while times < mpc.MAX_TIME:

            # get current state
            di = mpc.control_steer()
    
            ai = mpc.oa[0]
    
            # get new state
            mpc.state = mpc.update_state(mpc.state, ai, di)
            times = times + mpc.DT
    
            x.append(mpc.state.x)
            y.append(mpc.state.y)
            yaw.append(mpc.state.yaw)
            v.append(mpc.state.v)
            t.append(times)
            d.append(di)
            a.append(ai)
    
            #if mpc.show_animation:
            if False:
                plt.cla()
                #if ox is not None:
                #    plt.plot(ox, oy, "xr", label="MPC")
                plt.plot(mpc.road.cx, mpc.road.cy, "-r", label="course")
                plt.plot(x, y,  label="trajectory", linewidth=3)
                plt.plot(mpc.road.xref[0, :], mpc.road.xref[1, :], "xk", label="xref")
                mpc.plot_car(mpc.state.x, mpc.state.y ,mpc.state.yaw, steer=di)
                plt.axis("equal")
                plt.grid(True)
                plt.title("speed[km/h]:" + str(round(mpc.state.v*3.6, 2)) +
                          ", yaw:" + str(round(mpc.state.yaw,2)) + 
                          ", steer" + str(round(di,2)))
                plt.pause(0.0001)
            #time.sleep(0.1)
    
        return t, x, y, yaw, v, d, a
    t, x, y, yaw, v, d, a = do_simulation2(mpc)

if __name__ == '__main__':
    main()
