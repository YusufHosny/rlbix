import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class CubeRenderer:
    def __init__(self, cube):
        self.cube = cube
        self.fig = plt.figure()
        plt.ion()
        self.update()

    def update(self):
        plt.clf()
        self.ax = plt.axes(projection='3d')
        self.ax.set_facecolor('Cyan')
        colors = ['#FFFFFF', '#FFFF00', '#00FF00', '#0000FF', '#FFA500', '#FF0000'] # WYGBOR
        
        for channel, color in zip(self.cube.state, colors):
            for x, xdim in enumerate(channel):
                for y, ydim in enumerate(xdim):
                    for z, value in enumerate(ydim):
                        if value == 1: self.__draw_face(color, [x, y ,z])

        self.ax.axis('off')
        plt.show()
        plt.pause(.001)

    def interactive(self, time=None):
        if time == None:
            plt.ioff()
            plt.show()
        else:
            plt.pause(time)
        
    def __draw_face(self, color: str, position: list[int]):
        x, y, z = position
        if z == 0:
            xx, yy = np.meshgrid([x-1, x], [y-1, y])
            zz = np.ones((2,2)) * 0
            self.ax.plot_surface(xx, yy, zz, facecolor=color, edgecolor='k')
        elif z == 4:
            xx, yy = np.meshgrid([x-1, x], [y-1, y])
            zz = np.ones((2,2)) * 3
            self.ax.plot_surface(xx, yy, zz, facecolor=color, edgecolor='k')
        elif x == 0:
            zz, yy = np.meshgrid([z-1, z], [y-1, y])
            xx = np.ones((2,2)) * 0 
            self.ax.plot_surface(xx, yy, zz, facecolor=color, edgecolor='k')
        elif x == 4:
            zz, yy = np.meshgrid([z-1, z], [y-1, y])
            xx = np.ones((2,2)) * 3 
            self.ax.plot_surface(xx, yy, zz, facecolor=color, edgecolor='k')
        elif y == 0:
            xx, zz = np.meshgrid([x-1, x], [z-1, z])
            yy = np.ones((2,2)) * 0
            self.ax.plot_surface(xx, yy, zz, facecolor=color, edgecolor='k')
        elif y == 4:
            xx, zz = np.meshgrid([x-1, x], [z-1, z])
            yy = np.ones((2,2)) * 3
            self.ax.plot_surface(xx, yy, zz, facecolor=color, edgecolor='k')
        else:
            print('how did we get here')
        
        