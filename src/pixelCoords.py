class PixelCoords:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def __add__(self, other):
        return PixelCoords(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return PixelCoords(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return PixelCoords(self.x * other, self.y * other)

    def __floordiv__(self, other):
        return PixelCoords(self.x // other, self.y // other)

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"

    def __iter__(self):
        return iter((self.x, self.y))
    
    def distPixels(first, second):
        # print(first)
        # print(second)
        return (first.x - second.x) ** 2 + (first.y - second.y) ** 2

