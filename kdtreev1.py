import math
import numpy as np
import copy

class Node:
    def __init__(self,axis,cordinate,leftchild,righchild,pointindex):
        self.axis=axis #the index of axis ie:(x,y,z) -->(0,1,2)
        self.cordinate=cordinate # the cordinates of hyperhalfspace
        self.leftchild=leftchild
        self.rightchild = righchild
        self.pointindex=pointindex #  point index  of subtrees

    def is_leaf(self):
        if self.cordinate==None:
            return True
        else:
            return False
    def __str__(self):
        output=''
        output+='axis %d,'%self.axis
        if self.is_leaf():
            output+=' is leaf,'
        else:
            output+='the splitted plane cordinate: %0.3f'%self.cordinate
        output +='points: '
        output +=str(self.points.tolist())
        return output


def __build_kdTree(root,db,point_index,axis,leaf_size):
    """
    :param root: 根节点
    :param db:   点云数据
    :param point_index: 排序的键值
    :param axis: 沿aixs 排序
    :param leaf_size: 最小划分单位
    :return:
    """
    if root == None:
        root = Node(axis,None,None,None,point_index)

    if leaf_size>=len(point_index):
        return root

    sorted_index=np.argsort(db[point_index,axis])

    median_index =  len(sorted_index)//2
    left_point = db[sorted_index[median_index],axis]
    right_point = db[sorted_index[median_index+1],axis]
    root.cordinate=(left_point+right_point)*0.5

    root.leftchild =__build_kdTree(root.leftchild,db,sorted_index[:median_index],(axis+1)%db.shape[1],leaf_size)
    root.rightchild=__build_kdTree(root.rightchild,db,sorted_index[median_index:],(axis+1)%db.shape[1],leaf_size)
    return root

def build_kdTree(db,leaf_size=4):
    axis=0
    point_index = [i for i in range(db.shape[0])]
    root=None
    root=__build_kdTree(root,db,point_index,axis,leaf_size)
    return root






class DistIndex:
    def __init__(self, distance, index):
        self.distance = distance
        self.index = index

    def __lt__(self, other):
        return self.distance < other.distance




class KNNResult:
    def __init__(self,k):
        self.capacity=k
        self.count=0
        self.worst_distance=1e10
        self.distance_list=[]
        for i in range(self.capacity):
            self.distance_list.append(DistIndex(self.worst_distance,0))

    def worstdistance(self):
        return self.worst_distance
    def distancelsit(self):
        return self.distance_list

    def add_point(self,dis,index):
        if dis > self.worst_distance:
            return
        if self.count<self.capacity:
            self.count+=1
        #插入点的个数多余count时,插入点就放在capacity
        #即使超过count，也得插入,找出全局最小k个最邻近点值
        i= self.count-1
        #插入最小距离点
        while i>=1 and dis<self.distance_list[i-1].distance:
            self.distance_list[i]=copy.deepcopy(self.distance_list[i-1])
            i=i-1
        self.distance_list[i].distance=dis
        self.distance_list[i].index=index

        #当插入的
        self.worst_distance=self.distance_list[self.capacity-1].distance






def _search_knn(root:Node ,db:np.array,query_point:np.array, res:KNNResult):
    if(root == None):
        return False
    if(root.is_leaf()):
        points = db[root.pointindex,:]
        diff = np.linalg.norm(np.expand_dims(query_point,axis=0)-points,axis=1)
        for i in range(diff.shape[0]):
            res.add_point(diff[i],root.pointindex[i])
        return False

    if query_point[root.axis]<= root.cordinate:
        _search_knn(root.leftchild,db,query_point, res)
        if math.fabs(query_point[root.axis]-root.cordinate)<res.worstdistance():
            _search_knn(root.rightchild,db,query_point,res)

    else:
        _search_knn(root.rightchild,db,query_point,res)
        if math.fabs(query_point[root.axis]-root.cordinate)<res.worstdistance():
            _search_knn(root.leftchild,db,query_point,res)
    return False

def search_knn(db:np.array,query_point:np.array, k=10):
     res=KNNResult(10)
     kdTree= build_kdTree(db,leaf_size=4)
     _search_knn(kdTree,db,query_point,res)
     neartest_points=[]
     for i in range(k):
         point_Index=res.distance_list[i].index
         neartest_points.append(db[point_Index])
     neartest_points=np.asarray(neartest_points,dtype=np.float64)
     return neartest_points


def main():
    db_size = 64
    dim = 3
    leaf_size = 4
    k = 1
    db_np = np.random.randint(0,10,60).reshape(-1,3)
    print(db_np)
    res = search_knn(db_np,np.array([1,1,1]))
    print(res)
if __name__ == '__main__':
    main()