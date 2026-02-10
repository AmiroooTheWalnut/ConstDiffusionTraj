

class SubTraj:
    def __init__(self,initLat,initLon,cutLat,cutLon,points,isFromInit, isContinue):
        self.initLat=initLat# always shows the original lat/lon space to distinguish the starting point
        self.initLon=initLon
        self.points=points
        self.isFromInit=isFromInit
        self.cutLat=cutLat# always shows the starting location of a cut trajectory in normalized space
        self.cutLon=cutLon
        self.streetPoints=[]
        self.isContinue=isContinue
