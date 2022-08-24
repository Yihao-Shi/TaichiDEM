import numpy as np
import os


def MonitorDEM(time, partList, printNum, ascPath):
    print('---------------------', 'Writing DEM Monitor files ', printNum, '---------------------')
    if printNum == 0:
        if not os.path.exists(ascPath):
            os.mkdir(ascPath)

    filename = os.path.join(ascPath, 'MonitorDEM%06d.npz' % printNum)
    particleNum = np.ascontiguousarray(partList.particleNum.to_numpy()[None])
    pos_x = np.ascontiguousarray(partList.x.to_numpy()[0:partList.particleNum[None], 0])
    pos_y = np.ascontiguousarray(partList.x.to_numpy()[0:partList.particleNum[None], 1])
    pos_z = np.ascontiguousarray(partList.x.to_numpy()[0:partList.particleNum[None], 2])
    vel_x = np.ascontiguousarray(partList.v.to_numpy()[0:partList.particleNum[None], 0])
    vel_y = np.ascontiguousarray(partList.v.to_numpy()[0:partList.particleNum[None], 1])
    vel_z = np.ascontiguousarray(partList.v.to_numpy()[0:partList.particleNum[None], 2])
    w_x = np.ascontiguousarray(partList.w.to_numpy()[0:partList.particleNum[None], 0])
    w_y = np.ascontiguousarray(partList.w.to_numpy()[0:partList.particleNum[None], 1])
    w_z = np.ascontiguousarray(partList.w.to_numpy()[0:partList.particleNum[None], 2])

    np.savez(filename, particleNum=particleNum,
                       pos_x=pos_x, pos_y=pos_y, pos_z=pos_z, vel_x=vel_x, vel_y=vel_y, vel_z=vel_z, w_x=w_x, w_y=w_y, w_z=w_z, t=time)
