import time

import numpy as np
import matplotlib.pyplot as plt
import quaternion

import surface
import camera


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    scene = [
        surface.TexturedPolygon(surface.chessboard,np.array([[-100.0, -1, -100], [-100, -1, 100], [100, -1, 101], [100, -1, -100]]),
                        np.array([0.5, 0.5, 1])),
        surface.Sphere(np.array([0,0,0]),1)]

    c = camera.Camera((500, 500), 35, np.array([-4, 0, -2]), quaternion.quaternion(0, 0, 1, 0), 1)

    s = time.time()
    colours = c.render(scene, surface.default_sky, 4)
    e = time.time()
    print(e - s)
    plt.imshow(colours)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
