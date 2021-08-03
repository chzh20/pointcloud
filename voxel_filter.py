# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
from  pandas import  DataFrame
# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size,mode='centroid'):
    filtered_points = []

    #step1: get bounding box and divide the box by leaf_size
    min_data = point_cloud.min(axis=0)
    max_data = point_cloud.max(axis=0)
    D = (max_data-min_data)/leaf_size


    #step2: 计算点云所在的box的index
    points_x,points_y,points_z = point_cloud[:,0],point_cloud[:,1],point_cloud[:,2]

    #note : 向下取整
    h_x = np.floor((points_x - min_data[0]) / leaf_size)
    h_y = np.floor((points_y - min_data[1]) / leaf_size)
    h_z = np.floor((points_z - min_data[2]) / leaf_size)

    #print("h_x: ",h_x[:10,])
    h=np.array(np.floor(h_x + h_y *D[0] + h_z * D[0] *D[1]))

    #step3：根据点云所在下标排序,然后求相同下标的点平均值或者随机取点
    data = np.c_[h,point_cloud]
    data = data[data[:,0].argsort()]


    #随机取点
    if mode=='random':
        #filter_points=[]
        for i in range(data.shape[0]-1):
            if data[i][0]!= data[i+1][0]:
                filtered_points.append(data[i][1:])

        filtered_points.append(data[data.shape[0]-1][1:])

    #取平均点
    # #print(data[:,1:])
    if mode=='centroid':
        startindex = 0
        for i in range(data.shape[0] - 1):
            if (data[i][0] == data[i + 1][0]):
                continue
            else:
               # print(data[startindex:i+1][1:])
                temp_data=np.array(data[startindex:i+1,1:])
                filtered_points.append(temp_data.mean(axis=0))
                startindex = i + 1

        if startindex != len(data):
            filtered_points.append(np.mean(data[startindex:,1:], axis=0))






    # 屏蔽结束
   # print(h)
    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points[1:], dtype=np.float64)
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # 加载自己的点云文件
    file_name = "G:\\Project\\Point Cloud\\dataset\\modelnet40_normal_resampled\\car\\car_0005.txt"
    #point_cloud_pynt = PyntCloud.from_file(file_name)

    # 转成open3d能识别的格式
   # point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云
    point_cloud_raw = np.genfromtxt(file_name,delimiter=",")
    point_cloud_raw = DataFrame(point_cloud_raw[:,0:3])
    point_cloud_raw.columns=['x','y','z']
    point_cloud_pynt = PyntCloud(point_cloud_raw)


    point_cloud_o3d = point_cloud_pynt.to_instance('open3d',mesh=False)
    # 调用voxel滤波函数，实现滤波
    #std::vector<Eigen::Vector3d> with 460400 elements.
    print(point_cloud_o3d)
    print(np.asarray(point_cloud_o3d))
    print(type(point_cloud_o3d))
    filtered_cloud = voxel_filter(np.asarray(point_cloud_o3d.points),0.1)
    #print(filtered_cloud)
    #print(filtered_cloud)
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()
