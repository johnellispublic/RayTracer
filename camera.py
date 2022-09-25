import numpy as np
import quaternion


class Stack:
    def __init__(self):
        self.data = []

    def push(self, item):
        self.data.append(item)

    def pop(self):
        return self.data.pop()

    def peek(self):
        return self.data[-1]

    def is_empty(self):
        return len(self.data) == 0


class Camera:
    def __init__(self, res, FOV, location, axis=quaternion.quaternion(0,1, 0, 0), angle=0):
        self.res = res
        res_ratio = res[1] / res[0]
        self.FOV = FOV
        minx = np.tan(-FOV)
        maxx = -minx

        miny = maxx * res_ratio
        maxy = minx * res_ratio

        pixel_x, pixel_y = np.meshgrid(np.linspace(minx, maxx, res[0]), np.linspace(miny, maxy, res[1]))
        pixel_z = 1 + np.zeros(res)
        self.direction = np.swapaxes((pixel_x, pixel_y, pixel_z),0,2)

        rotation_quaternion = np.cos(angle / 2) + axis * np.sin(angle / 2)
        self.direction = quaternion.rotate_vectors(rotation_quaternion, self.direction)

        self.starts = np.zeros((res[0], res[1], 3)) + location

    @staticmethod
    def hits_sky(ray_starts, ray_ends, scene):
        hits_sky = np.zeros(np.shape(ray_starts)[:-1])
        for surface in scene:
            hits_sky[(surface.get_distance(ray_starts, ray_ends) == np.inf)] = 1

        return hits_sky == 1

    def render(self, scene, sky, reflection_count=4):
        history = np.array([[Stack() for _ in range(self.res[0])] for _ in range(self.res[1])])

        ray_starts = np.reshape(self.starts, (-1, 3))
        ray_direction = np.reshape(self.direction, (-1, 3))
        ray_direction /= np.sqrt((ray_direction*ray_direction).sum(1))[:, np.newaxis]
        ray_index = np.reshape(np.array([[(i, j) for i in range(self.res[0])] for j in range(self.res[1])]), (-1, 2))

        for _ in range(reflection_count):
            new_ray_starts = np.zeros(np.shape(ray_starts))
            new_ray_direction = np.zeros(np.shape(ray_starts))
            print(len(new_ray_starts))
            distances = np.zeros(np.shape(ray_starts)[:-1]) + np.inf
            surfaces = np.ndarray(np.shape(ray_starts)[:-1], dtype=object)
            for surface in scene:
                new_distances = surface.get_distance(ray_starts, ray_direction)
                surfaces[(new_distances < distances)] = surface
                distances[(new_distances < distances)] = new_distances[(new_distances < distances)]

            for surface in scene:
                surface_is_surface = (surfaces == surface)

                new_ray_starts[(surface_is_surface)], new_ray_direction[(surface_is_surface)] = surface.reflect(
                    ray_starts[(surface_is_surface)],
                    ray_direction[(surface_is_surface)],
                    distances[(surface_is_surface)])

            for i in range(len(surfaces)):
                if surfaces[i] is not None:
                    history[ray_index[i][0], ray_index[i][1]].push((surfaces[i], new_ray_starts[i], new_ray_direction[i]))
                else:
                    history[ray_index[i][0], ray_index[i][1]].push((sky, ray_starts[i], ray_direction[i]))

            ray_starts = new_ray_starts[(surfaces != None)]
            ray_direction = new_ray_direction[(surfaces != None)]
            ray_index = ray_index[(surfaces != None)]

        hits_sky = self.hits_sky(ray_starts, ray_direction, scene)

        for index, starts, direction in zip(ray_index[(hits_sky)], ray_starts[(hits_sky)], ray_direction[(hits_sky)]):
            history[index[0], index[1]].push((sky, starts, direction))

        colour = []
        for row in history:
            colour.append([])
            for stack in row:
                if type(stack.peek()) == tuple and stack.peek()[0] is sky:
                    colour[-1].append(sky(*stack.pop()[1:]))
                else:
                    colour[-1].append(np.array([0, 0, 0]))
                    continue

                while not stack.is_empty():
                    surface, start, direction = stack.pop()
                    colour[-1][-1] = surface.modify_colour(colour[-1][-1], start, direction)

        colour = np.array(colour)
        colour[(colour < 0)] = 0
        colour[(colour > 1)] = 1

        return np.array(colour)
