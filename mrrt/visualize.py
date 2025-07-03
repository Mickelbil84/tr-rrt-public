import time

import trimesh
import numpy as np
# import pybullet as p
# import pybullet_data
from spatialmath import SE3

from .sdf import *


class BulletVisualization(object):
    def __init__(self, gui=True):
        self.gui = gui
        if gui:
            self.pc = p.connect(p.GUI)  # physics client
        else:
            return
            self.pc = p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        # p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)

        # self.default_plane = p.loadURDF("plane.urdf")
        self.objects = {}

    def add_urdf(self, path: str, name: str):
        if not self.gui:
            return
        self.objects[name] = p.loadURDF(path)

    def set_object_configuration(self, name: str, configuration: SE3):
        if not self.gui:
            return
        pos = list(configuration.t)
        orn = p.getQuaternionFromEuler(list(configuration.rpy()))
        p.resetBasePositionAndOrientation(self.objects[name], pos, orn)

    def set_object_color(self, name: str, color: list):
        if not self.gui:
            return
        p.changeVisualShape(
            self.objects[name], -1, rgbaColor=color, physicsClientId=self.pc)

    def add_debug_line(self, q1: SE3, q2: SE3, color: list, duration=0):
        if not self.gui:
            return
        p.addUserDebugLine(
            q1.t, q2.t,
            lineColorRGB=color, lineWidth=1 if color[2]==1 else 0.2, lifeTime=duration, physicsClientId=self.pc
        )

    def add_debug_line_from_xyz(self, q1: tuple, q2: tuple, color: list, duration=0):
        if not self.gui:
            return
        p.addUserDebugLine(q1, q2, lineColorRGB=color, lineWidth=1 if color[2]==1 else 0.2, lifeTime=duration, physicsClientId=self.pc)


    def removeAllUserDebugItems(self):
        p.removeAllUserDebugItems()

    def add_debug_point(self, q: SE3, size: float = 0.02, duration=0, color=None):
        if not self.gui:
            return
        q_r = q * SE3(size, 0, 0)
        q_g = q * SE3(0, size, 0)
        q_b = q * SE3(0, 0, size)
        x_color = [1, 0, 0] if color is None else color
        y_color = [0, 1, 0] if color is None else color
        z_color = [0, 0, 1] if color is None else color
        self.add_debug_line(q, q_r, x_color, duration=duration)
        self.add_debug_line(q, q_g, y_color, duration=duration)
        self.add_debug_line(q, q_b, z_color, duration=duration)

    def step(self):
        if not self.gui:
            return
        p.stepSimulation()


def update_rrt_visualization(bv, q, q_, color, update=True, duration=0):
    if not bv.gui:
            return
    if type(q) is not SE3:
        q = xyzrpy_2_SE3(q)
    if type(q_) is not SE3:
        q_ = xyzrpy_2_SE3(q_)

    bv.add_debug_line(q, q_, color, duration)
    if update:
        bv.set_object_configuration("m2", q_)
        bv.step()