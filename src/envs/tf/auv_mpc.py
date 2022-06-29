from pygame import init
from sympy import *
import numpy as np

# from envs.tf.mpc import control

class KineticModel:
    def __init__(self) -> None:
        self.posiion = Matrix(np.zeros(6))
        self.velocities = Matrix(np.zeros(6))
        # self.control_values = Matrix(np.zeros(8))


        # Kinetic model parameters
        # x = symbols('x')
        # y = symbols('y')
        # z = symbols('z')
        th = symbols('th')
        phi = symbols('phi')
        psi = symbols('psi')
        # u = symbols('u')
        # v = symbols('v')
        # w = symbols('w')
        # p = symbols('p')
        # q = symbols('q')
        # r = symbols('r')


        zer_matrix = Matrix(np.zeros((3,3)))
        Rth = expand(Matrix([[cos(psi)*cos(th), -sin(psi)*cos(phi) + cos(psi)*sin(phi)*sin(th), sin(psi)*sin(phi) + cos(psi)*cos(phi)*sin(th)],
            [sin(psi)*cos(th), cos(psi)*cos(phi) + sin(psi)*sin(phi)*sin(th), -cos(psi)*sin(phi) + sin(psi)*sin(th)*cos(phi)],
            [-sin(th), cos(th)*sin(phi), cos(th)*cos(phi)]]))

        Tth = expand(Matrix([[1, sin(phi)*tan(th), cos(phi)*tan(th)],
                        [0, cos(th), -sin(th)],
                        [0, sin(phi)/cos(th), cos(phi)/cos(th)]]))

        self.Jnu = expand(Matrix([[Rth, zer_matrix],
                            [zer_matrix, Tth]]))

        
        m = 11.5
        g = 9.82
        mg = m*g
        ro = 1000
        vol = 0.0114
        zg = 0.02
        rgv = ro*g*vol

        Ix = 0.16
        Iy = 0.16
        Iz = 0.16

        # added mass
        Xdu = -5.5
        Ydv = -12.7
        Zdw = -14.57
        Kdp = -0.12
        Mdq = -0.12
        Ldr = -0.12

        # damping coefficients. from heavy
        #linear
        Xu = -4.03
        Yv = -6.22
        Zw = -5.18
        Kp = -0.07
        Mq = -0.07
        Lr = -0.07
        #quadratic
        Xuu = -18.18
        Yvv = -21.66
        Zww = -36.99
        Kpp = -1.55
        Mqq = -1.55
        Lrr = -1.55
        
        # Rigid body matrix
        Mrb = expand(Matrix([[m, 0, 0, 0, m*zg, 0],
                    [0, m, 0, -m*zg, 0, 0],
                    [0, 0, m, 0, 0, 0],
                    [0, -m*zg, 0, Ix, 0, 0],
                    [m*zg, 0, 0, 0, Iy, 0],
                    [0, 0, 0, 0, 0, Iz]]))

        Ma = expand(Matrix([[Xdu, 0, 0, 0, 0, 0],
                            [0, Ydv, 0, 0, 0, 0],
                            [0, 0, Zdw, 0, 0, 0],
                            [0, 0, 0, Kdp, 0, 0],
                            [0, 0, 0, 0, Mdq, 0],
                            [0, 0, 0, 0, 0, Ldr]]))
        self.M = Ma + Mrb

        # Damping matrix
        self.D = expand(-Matrix([[Xuu + Xu, 0, 0, 0, 0, 0],
                    [0, Yvv + Yv, 0, 0, 0, 0],
                    [0, 0, Zww + Zw, 0, 0, 0],
                    [0, 0, 0, Kpp + Kp, 0, 0],
                    [0, 0, 0, 0, Mqq + Mq, 0],
                    [0, 0, 0, 0, 0, Lrr + Lr]]))
        
        # Hydrostatic forces
        self.G = expand(Matrix([[-(mg - rgv) * sin(th)],
                    [(mg - rgv) * cos(th) * sin(phi)],
                    [(mg - rgv) * cos(th) * cos(phi)],
                    [-zg*mg* cos(th) * sin(phi)],
                    [-zg*mg* sin(th)],
                    [0]]))

        # Thrusts configuration
        self.Thrusts = Matrix([[0.707, 0.707, -0.707, -0.707, 0, 0, 0, 0],
                    [-0.707, 0.707, -0.707, 0.707, 0, 0, 0, 0],
                    [0, 0, 0, 0, -1, 1, 1, -1],
                    [0.06, -0.06, 0.06, -0.06, -0.218, -0.218, 0.218, 0.218],
                    [0.06, 0.06, -0.06, -0.06, 0.12, -0.12, 0.12, -0.12],
                    [-0.1888, 0.1888, 0.1888, -0.1888, 0, 0, 0, 0]])

    def get_velocities(self, accs, control_values, agent_orientation):

        self.G = self.G.subs("phi", 0).subs("th", 0).subs("psi", agent_orientation)
        self.control_values = control_values
        # print("\n",self.Jnu)
        # print("\n",self.Jnu)
        invD = expand(self.D.inv())
        K = expand(Matrix(np.diag(np.full(8, 1/40))))
        # Des = Matrix(np.diag([desired[0], desired[1], 0, 0, 0, des]))
        # invK = expand(K.inv())

        # invThrusts = self.Thrusts.inv()
        # u = invK * invThrusts * tau
        control_values = Matrix(control_values)
        accs = Matrix(accs)
        # print(control_values.shape)
        tau =  -20000 * self.Thrusts * K * control_values
        # print("\ntau: ", tau.shape)
        self.velocities = invD * (tau - self.G - self.M * accs)
        # print("\n", self.velocities,"\n")
        return self.velocities

    def get_position(self, agent_orientation):
        self.Jnu = self.Jnu.subs("phi", 0).subs("th", 0).subs("psi", agent_orientation)

        self.posiion = self.Jnu * self.velocities
        return self.posiion

    def set_control(self, control_values):
        pass


    # X = symbols('X')
    # Y = symbols('Y')
    # Z = symbols('Z')
    # K = symbols('K')
    # M = symbols('M')
    # L = symbols('L')

    # tau = symbols('tau')



    # print(Jnu)

    # The vehicle velocity is small. Therefore Coriolis, centripetal forces and non-
    # linear damping can be neglected.
    # • Underwater currents are constant or non-existent. Therefore the equation is
    # given in terms of the vehicle velocity ν.
    # • Roll and pitch motions are below 10 degrees, and the ROV is assumed to be
    # neutrally buoyant with the CB straight over the CG. Then linearization can
    # be used on the restoring forces using G.


    # Crb = expand(Matrix([[0, 0, 0, m*zg*r, m*w, -m*v],
    #                 [0, 0, 0, -m*w, 0, m*u],
    #                 [0, 0, 0, m*v, -m*(zg*q+u), 0],
    #                 [-m*zg*r, m*w, -m*v, 0, Iz*q, -Iy*p],
    #                 [-m*w, -m*zg*r, 0, -Iz*q, 0, Ix*p],
    #                 [m*v, -m*u, 0, Iy*q, -Ix*p, 0]]))

    # Ca = expand(Matrix([0, 0, 0, 0, -Zw*w, Yv*v],
    #                     [0, 0, 0, Zw*w, 0, -Xu*u],
    #                     [0, 0, 0, -Yv*v, Xu*u, 0],
    #                     [0, -Zw*w, Yv*v, 0, -Lr*r, Mq*q],
    #                     [Zw*w, 0, -Xu*u, Lr*r, 0, -Kp*p],
    #                     [-Yv*v, Xu*u, 0, -Mq*q, Kp*p, 0]))




    # η̇ = J(η)ν
    # M ν̇ + Dν + G(η) = τ + J > (η)b
    # ḃ = −T −1
    # b b + ω b

    # nu = [x, y, z, phi, th, psi]
    # vel =[u, v, w, p, q, r]

    # tau = [X, Y, Z, K, M, L]
    # print(transpose(Jnu))
    # dnu = Jnu * velocities

    # h = symbols('h')

    # print(Rth)