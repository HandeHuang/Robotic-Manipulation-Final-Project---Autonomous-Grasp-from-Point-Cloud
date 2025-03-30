# Robotic Manipulation Final Project - Autonomous Grasp from Point Cloud
 ROB 498/599 Intro to Robotic Manipulation final project.


# 🦾 Antipodal Grasping with Point Clouds

This project implements an automated grasping pipeline in a simulated PyBullet environment. It uses point cloud data to detect objects, compute antipodal grasps, and control a robot to remove objects from the scene.

---

## Demo:

![Grasp Demo](demo.gif)


---

## 🚀 Pipeline Overview

1. **Initialize Simulation**  
   Set up the environment.

2. **Capture Scene**  
   Retrieve point cloud data from virtual cameras.

3. **Segment Objects**  
   Use DBSCAN clustering to identify individual objects.

4. **Compute Grasp Poses**  
   For each object, find a grasp point near the centroid and compute grasp orientation (`theta`) based on surface normals.

5. **Execute Grasp**  
   Move the robot to perform the grasp and drop the object into a bin.

6. **Repeat**  
   Loop until the scene is cleared.

---

## 🧰 Dependencies

- `numpy`  
- `open3d`  
- `pybullet`  
- `matplotlib`  
- Custom helper: `assignment_6_helper.py`

