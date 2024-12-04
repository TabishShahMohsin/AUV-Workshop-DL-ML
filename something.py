class rectangle():
    def __init__(self,l,b):
        self.length=l
        self.breadth=b
    def area(self):
        return self.length*self.breadth
    def parameter(self):
        return 2*(self.length+self.breadth)
plot=rectangle(5,3)
print(plot.area())