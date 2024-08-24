# TODO retrieve model data from data folder, train with requested options, then run model with gui for inference

from render.render_cube import CubeRenderer
from environment.cube import Cube
import time


# Create cube and render it, apply right rotation every second
cube = Cube()
renderer = CubeRenderer(cube)

while True:
    cube.right()
    renderer.update()
    renderer.interactive(1)