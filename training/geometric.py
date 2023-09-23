class PointInt:
    EX: 'PointInt'
    EY: 'PointInt'
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    def __add__(self, lhs: 'PointInt'):
        return PointInt(self.x + lhs.x, self.y + lhs.y)
    def __sub__(self, lhs: 'PointInt'):
        return PointInt(self.x - lhs.x, self.y - lhs.y)
    def __mul__(self, lhs: int | float):
        return PointInt(int(self.x * lhs), int(self.y * lhs))
    def __iter__(self):
        yield self.x
        yield self.y
PointInt.EX = PointInt(1, 0)
PointInt.EY = PointInt(0, 1)
    
class RectInt:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    @staticmethod
    def from_points(point1: PointInt, point2: PointInt):
        return RectInt(point1.x, point1.y, point2.x - point1.x, point2.y - point1.y)
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.w
        yield self.h
    def to_points(self):
        return PointInt(self.x, self.y), PointInt(self.x + self.w, self.y + self.h)