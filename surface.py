import numpy as np
import itertools as it


def normalise_sq(vec3):
    return (vec3*vec3).sum(1)


def default_sky(position, direction):
    return np.sin(2 * (direction * 0.5 + 1))


# noinspection PyUnresolvedReferences
class Surface:
    def __init__(self, colour):
        self.roughness = 0
        self.colour = colour

    def get_intersect(self, ray_starts, ray_direction, distance=None):
        if distance is None:
            distance = self.get_distance(ray_starts, ray_direction)
        return distance[:, np.newaxis] * ray_direction + ray_starts

    def get_distance(self, ray_starts, ray_direction):
        pass

    def intersects(self, ray_starts, ray_direction, distance=None):
        if distance is None:
            distance = self.get_distance(ray_starts, ray_direction)
        return 0 < distance < np.inf

    def get_normal(self, position):
        pass

    def reflect(self, ray_starts, ray_direction, distance=None):
        if len(ray_starts) == 0:
            return np.ndarray((0, 3)), np.ndarray((0, 3))
        intersections = self.get_intersect(ray_starts, ray_direction, distance)
        normal = self.get_normal(intersections)

        central_out = ray_direction - 2 * (ray_direction*normal).sum(1)[:, np.newaxis] * normal

        direction = central_out / (central_out * central_out).sum(1)[:, np.newaxis]

        return intersections + direction * 0.1, direction

    def modify_colour(self, colour, position=None, direction=None):
        return colour * self.colour


class Polygon(Surface):
    def __init__(self, points, colour):
        super().__init__(colour)
        self.points = points
        self.v = points[1] - points[0]
        self.u = points[2] - points[0]
        self.normal = np.cross(self.u, self.v)
        self.normal /= np.linalg.norm(self.normal)

        self.flatten_matrix = np.linalg.inv([self.u, self.v, -self.normal])

        self.flattened_points = self.points @ self.flatten_matrix

    def __get_edges(self):
        return it.pairwise(it.chain(self.flattened_points, [self.flattened_points[0]]))

    def get_distance(self, ray_starts, ray_directions):
        t = (self.points[0] - ray_starts).dot(self.normal) / (ray_directions.dot(self.normal))

        intersects = t[:, np.newaxis] * ray_directions + ray_starts
        intersects = intersects @ self.flatten_matrix
        edge_count = np.zeros(np.shape(intersects)[:-1])

        for point1, point2 in self.__get_edges():
            if point1[0] == point2[0]:
                edge_count[((point1[0] - intersects[:, 0]) * (point2[0] - intersects[:, 0]) < 0)] += 1
                continue

            t2 = (intersects[:, 0] - point1[0]) / (point2[0] - point1[0])
            d = point1[1] - intersects[:, 1] + t2 * (point2[1] - point1[1])

            edge_count[((0 < t2) & (t2 < 1) & (d > 0))] += 1

        t[(edge_count % 2 == 0)] = np.inf
        t[(t < 0)] = np.inf

        return t

    def get_normal(self, position):
        return self.normal


class Sphere(Surface):
    def __init__(self, centre, radius, colour=np.array([0.5, 0.5, 0.5])):
        super().__init__(colour)

        self.centre = centre
        self.radius = radius

    def get_distance(self, ray_starts, ray_direction):
        a = 1
        b = 2*(ray_direction*(ray_starts - self.centre)).sum(1)
        c = normalise_sq(ray_starts - self.centre) - self.radius**2

        discr = b**2 - 4*a*c
        discr_sqrt = np.sqrt(discr)

        t = np.zeros(np.shape(ray_starts)[:-1])

        t[(-b + discr_sqrt > 0)] = ((-b+discr_sqrt)/2)[(-b + discr_sqrt > 0)]
        t[(-b - discr_sqrt > 0)] = ((-b - discr_sqrt)/2)[(-b - discr_sqrt > 0)]
        t[(discr < 0)] = np.inf

        return t

    def get_normal(self, position):
        normal = position - self.centre
        return normal / (normal*normal).sum(1)[:, np.newaxis]


class TexturedSurface(Surface):
    def __init__(self, texture=lambda pos: np.array([1, 1, 1]), *args, **kwargs):
        self.texture = texture
        super().__init__(*args, **kwargs)

    def modify_colour(self, colour, position=None, direction=None):
        return colour * (self.colour + self.texture(position)) / 2


class TexturedPolygon(TexturedSurface, Polygon):
    pass


def chessboard(pos):
    return 1 - np.sum(np.floor(pos % 2)) % 2