import os
from pyevtk.hl import pointsToVTK
import numpy as np


def WriteFileVTK_DEM(partList, printNum, vtkPath):
    print('---------------------', 'Writing DEM Graphic files ', printNum, '---------------------')

    if printNum == 0:
        if not os.path.exists(vtkPath):
            os.mkdir(vtkPath)

    pos_x = np.ascontiguousarray(partList.x.to_numpy()[0:partList.particleNum[None], 0])
    pos_y = np.ascontiguousarray(partList.x.to_numpy()[0:partList.particleNum[None], 1])
    pos_z = np.ascontiguousarray(partList.x.to_numpy()[0:partList.particleNum[None], 2])
    ID = np.ascontiguousarray(partList.ID.to_numpy()[0:partList.particleNum[None]])
    shapeType = np.ascontiguousarray(partList.shapeType.to_numpy()[0:partList.particleNum[None]])
    group = np.ascontiguousarray(partList.group.to_numpy()[0:partList.particleNum[None]])
    rad = np.ascontiguousarray(partList.rad.to_numpy()[0:partList.particleNum[None]])
    vel_x = np.ascontiguousarray(partList.v.to_numpy()[0:partList.particleNum[None], 0])
    vel_y = np.ascontiguousarray(partList.v.to_numpy()[0:partList.particleNum[None], 1])
    vel_z = np.ascontiguousarray(partList.v.to_numpy()[0:partList.particleNum[None], 2])
    w_x = np.ascontiguousarray(partList.w.to_numpy()[0:partList.particleNum[None], 0])
    w_y = np.ascontiguousarray(partList.w.to_numpy()[0:partList.particleNum[None], 1])
    w_z = np.ascontiguousarray(partList.w.to_numpy()[0:partList.particleNum[None], 2])
    w = np.sqrt(w_x ** 2 + w_y ** 2+ w_z ** 2)
    pointsToVTK(vtkPath+f'/GraphicDEM{printNum:06d}', pos_x, pos_y, pos_z, data={'ID': ID, 'shapeType': shapeType, 'group': group, 'radius': rad, 'vel_x':vel_x, 'vel_y':vel_y, 'vel_z':vel_z, 'w':w})

