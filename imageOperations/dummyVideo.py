import cv2 

class dummyVideo:
    def __init__(self, filename):
        self.image = cv2.imread(filename)
    
    def read(self):
        return (True, self.image)
    
    def isOpened(self):
        return True

    def release(self):
        pass
    