import numpy as np
import pybullet as p
import open3d as o3d
import assignment_6_helper as helper


# def get_antipodal(pcd):
#     """
#     function to compute antipodal grasp given point cloud pcd
#     :param pcd: point cloud in open3d format (converted to numpy below)
#     :return: gripper pose (4, ) numpy array of gripper pose (x, y, z, theta)
#     """
#     # convert pcd to numpy arrays of points and normals
#     pc_points = np.asarray(pcd.points)
#     pc_normals = np.asarray(pcd.normals)

#     # ------------------------------------------------
#     # FILL WITH YOUR CODE

#     # gripper orientation - replace 0. with your calculations
#     theta = 0.
#     # gripper pose: (x, y, z, theta) - replace 0. with your calculations
#     gripper_pose = np.array([-0.04999192, -0.00671114,  0.03313947, theta])
#     # ------------------------------------------------

#     return gripper_pose

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def find_object_centers(pcd, distance_threshold=0.05, min_points=50):
    """
    Find the centers of separate objects in the point cloud using DBSCAN clustering
    and return both the centroids and the segmented point clouds.

    Additionally, assigns unique colors to each segmented point cloud for visualization.

    :param pcd: open3d.geometry.PointCloud
    :param distance_threshold: float, the epsilon parameter for DBSCAN
    :param min_points: int, minimum number of points to form a cluster
    :return: 
        centroids: list of centroids as numpy arrays
        segmented_pcds: list of open3d.geometry.PointCloud objects for each cluster
    """
    # Optional Preprocessing: Downsample the point cloud for faster processing
    down_pcd = pcd.voxel_down_sample(voxel_size=0.005)  # Adjust voxel_size as needed

    # Optional Preprocessing: Remove statistical outliers
    cl, ind = down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    clean_pcd = down_pcd.select_by_index(ind)

    # Perform DBSCAN clustering
    labels = np.array(clean_pcd.cluster_dbscan(eps=distance_threshold, min_points=min_points, print_progress=True))

    if labels.size == 0:
        print("No clusters found.")
        return [], []

    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters")

    if max_label + 1 < 2:
        print("Less than two clusters found. Adjust DBSCAN parameters.")
        # Compute centroid of the entire point cloud
        centroid = np.asarray(pcd.points).mean(axis=0)
        return [centroid], [pcd]

    # Assume the two largest clusters are the objects
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]  # Descending order
    top_labels = unique_labels[sorted_indices[:2]]  # Top two clusters

    centroids = []
    segmented_pcds = []
    for label in top_labels:
        cluster_indices = np.where(labels == label)[0]
        cluster_points = np.asarray(clean_pcd.points)[cluster_indices]
        centroid = cluster_points.mean(axis=0)
        centroids.append(centroid)
        print(f"Centroid of cluster {label}: {centroid}")

        # Create a new PointCloud for the cluster
        cluster_pcd = clean_pcd.select_by_index(cluster_indices.tolist())

        # Assign a unique color to each cluster for visualization
        color = plt.get_cmap("tab20")(label % 20)[:3]  # Cycle through 20 colors
        cluster_pcd.paint_uniform_color(color)

        segmented_pcds.append(cluster_pcd)

    return centroids, segmented_pcds

def get_antipodal(pcd, delta_z=0.003, elevation=-0.01):
    """
    Compute antipodal grasp poses for separate objects in the point cloud and return
    both the gripper poses and the segmented point clouds.

    For each object, aligns the gripper's x, y, z with the centroid's x, y, z
    (elevated by a specified amount) and calculates theta based on the normals.

    :param pcd: open3d.geometry.PointCloud
    :param delta_z: float, tolerance for selecting points near the plane z = centroid_z
    :param elevation: float, height to elevate the gripper above the centroid
    :return: 
        gripper_poses: list of numpy arrays [x, y, z, theta] for each grasp
        segmented_pcds: list of open3d.geometry.PointCloud objects for each object
    """
    # Convert pcd to numpy arrays of points and normals
    pc_points = np.asarray(pcd.points)
    pc_normals = np.asarray(pcd.normals)

    # ------------------------------------------------
    # FILL WITH YOUR CODE

    # gripper orientation - replace 0. with your calculations
    theta = 0.
    # gripper pose: (x, y, z, theta) - replace 0. with your calculations
    gripper_pose = np.array([-0.04999192, -0.00671114,  0.03313947, theta])

    # # Handle empty point cloud case
    # if pc_points.shape[0] == 0:
    #     # No objects: return a default pose and empty list
    #     return [np.array([0., 0., 0.3, 0.])], []

    # Step 1: Find centers and segmented point clouds of the objects
    centroids, segmented_pcds = find_object_centers(pcd)

    # if not centroids:
    #     # If no centroids found, return a default grasp pose
    #     return [np.array([0., 0., 0.3, 0.])], []

    gripper_poses = []
    for centroid, obj_pcd in zip(centroids, segmented_pcds):
        # Step 2: For each object, find the point within a range around z = centroid_z closest to the centroid
        object_points = np.asarray(obj_pcd.points)
        object_normals = np.asarray(obj_pcd.normals)

        if object_points.shape[0] == 0:
            print("No points found in segmented point cloud. Skipping.")
            continue

        # Define the plane z = centroid_z with a tolerance (range)
        centroid_z = centroid[2]
        lower_bound = centroid_z - delta_z
        upper_bound = centroid_z + delta_z

        # Create a mask for points within the z-range
        mask = (object_points[:, 2] >= lower_bound) & (object_points[:, 2] <= upper_bound)
        points_in_range = object_points[mask]
        normals_in_range = object_normals[mask]

        if points_in_range.shape[0] > 0:
            # Compute Euclidean distances in x and y from the centroid
            distances_xy = np.linalg.norm(points_in_range[:, :2] - centroid[:2], axis=1)
            closest_idx = np.argmin(distances_xy)
            grasp_point = points_in_range[closest_idx]
            grasp_normal = normals_in_range[closest_idx]
            print(f"Found point within z-range for centroid {centroid}: {grasp_point}")
        else:
            # If no points are within the z-range, find the point closest to the centroid in 3D space
            distances = np.linalg.norm(object_points - centroid, axis=1)
            closest_idx = np.argmin(distances)
            grasp_point = object_points[closest_idx]
            grasp_normal = object_normals[closest_idx]
            print(f"No points within z-range. Selected closest point to centroid {centroid}: {grasp_point}")

        # Compute theta: the angle between the grasp normal and the x-axis
        theta = np.arctan2(grasp_normal[1], grasp_normal[0])

        # Set the gripper's x, y, z to match the centroid's x, y, z, elevated by 'elevation'
        gripper_x = centroid[0]
        gripper_y = centroid[1]
        gripper_z = centroid[2] + elevation # Elevate above the centroid, edited to negative

        gripper_pose = np.array([gripper_x, gripper_y, gripper_z, theta])
        gripper_poses.append(gripper_pose)
        print(f"Gripper pose aligned with centroid {centroid}: {gripper_pose}")

    return gripper_poses[0]


def main(n_tries=5):
    # Initialize the world
    world = helper.World()

    # start grasping loop
    # number of tries for grasping
    for i in range(n_tries):
        # get point cloud from cameras in the world
        pcd = world.get_point_cloud()
        # check point cloud to see if there are still objects to remove
        finish_flag = helper.check_pc(pcd)
        if finish_flag:  # if no more objects -- done!
            print('===============')
            print('Scene cleared')
            print('===============')
            break
        # visualize the point cloud from the scene
        helper.draw_pc(pcd)
        # compute antipodal grasp
        gripper_pose = get_antipodal(pcd)
        # send command to robot to execute
        print("LOOK HERE:",gripper_pose)
        robot_command = world.grasp(gripper_pose)
        # robot drops object to the side
        world.drop_in_bin(robot_command)
        # robot goes to initial configuration and prepares for next grasp
        world.home_arm()
        # go back to the top!

    # terminate simulation environment once you're done!
    p.disconnect()
    return finish_flag


if __name__ == "__main__":
    flag = main()
