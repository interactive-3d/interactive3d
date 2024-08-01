import copy
import open3d as o3d
import numpy as np


def fps(xyz, npoint): 

    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    if len(xyz.shape) == 2:
        has_b_dim = False
        xyz = xyz[None, ...]
    else:
        has_b_dim = True

    # xyz = xyz.transpose(2,1)
    B, N, C = xyz.shape
    
    centroids = np.zeros((B, npoint), dtype=np.int64)
    distance = np.ones((B, N)) * 1e10

    batch_indices = np.arange(B, dtype=np.int64)

    #farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    
    barycenter = np.sum(xyz, axis=1)
    barycenter = barycenter/xyz.shape[1]
    barycenter = barycenter.reshape(B, 1, 3)

    dist = np.sum((xyz - barycenter) ** 2, -1)
    farthest = np.max(dist, axis=1)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].reshape(B, 1, 3)
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.max(distance, -1)[1]
    
    if not has_b_dim:
        centroids = centroids[0]
    return centroids

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def draw(*geo):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(axis_pcd)
    [vis.add_geometry(g) for g in geo]
    vis.run()  # user picks points
    vis.destroy_window()

def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis = o3d.visualization.VisualizerWithVertexSelection()
    # vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    # vis.add_geometry(axis_pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    # return vis.get_cropped_geometry()
    return vis.get_picked_points()


def demo_manual_registration():
    print("Demo for manual ICP")
    pcd_data = o3d.data.DemoICPPointClouds()
    source = o3d.io.read_point_cloud(pcd_data.paths[0])
    target = o3d.io.read_point_cloud(pcd_data.paths[2])
    print("Visualization of two point clouds before manual alignment")
    draw_registration_result(source, target, np.identity(4))

    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source)
    picked_id_target = pick_points(target)
    assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
    assert (len(picked_id_source) == len(picked_id_target))
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target,
                                            o3d.utility.Vector2iVector(corr))

    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    threshold = 0.03  # 3cm distance threshold
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    draw_registration_result(source, target, reg_p2p.transformation)
    print("")


def transform_point_cloud_around_aabb_center(pcd, transformation):
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    aabb_center = aabb.get_center()

    to_origin = np.eye(4)
    to_origin[0:3, 3] = -aabb_center

    back_to_original = np.eye(4)
    back_to_original[0:3, 3] = aabb_center

    composed_transform = np.matmul(np.matmul(back_to_original, transformation), to_origin)
    transformed_pcd = pcd.transform(composed_transform)
    transformed_obb = transformed_pcd.get_oriented_bounding_box()
    transformed_obb.color = (0, 1, 0)

    return transformed_pcd, aabb, transformed_obb


class PointCloudTransformer:
    def __init__(self, pcd, pc_non_edit):
        self.pcd = pcd
        self.pc_non_edit = pc_non_edit
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        self.view_axis = np.array([0, 0, -1])  # Start with Z-axis (negative because of camera's default view)

    def reset_view(self):
        ctl = self.vis.get_view_control()
        print(self.view_axis)
        ctl.set_front(self.view_axis)
        # ctl.set_up([0, 1, 0])

    def set_axis_view_x(self, vis, action, mods):
        if action == 0:
            return False
        if np.array_equal(self.view_axis, [1, 0, 0]):
            self.view_axis = [-1, 0, 0]
            print('-X')
        else:
            self.view_axis = [1, 0, 0]
            print(' X')
        # self.reset_view()
        return False  # No update to geometry, just view

    def set_axis_view_y(self, vis, action, mods):
        if action == 0:
            return False
        if np.array_equal(self.view_axis, [0, 1, 0]):
            self.view_axis = [0, -1, 0]
            print('-Y')
        else:
            self.view_axis = [0, 1, 0]
            print(' Y')
        # self.reset_view()
        return False  # No update to geometry, just view

    def set_axis_view_z(self, vis, action, mods):
        if action == 0:
            return False
        if np.array_equal(self.view_axis, [0, 0, 1]):
            self.view_axis = [0, 0, -1]
            print('-Z')
        else:
            self.view_axis = [0, 0, 1]
            print(' Z')
        # self.reset_view()
        return False  # No update to geometry, just view

    def translate_view_plane(self, translation, action, mods):
        if action == 0:
            return False
        # Get orthogonal plane to view axis
        # orth_plane = np.outer(self.view_axis, self.view_axis)
        # translation = np.eye(3) - orth_plane.dot(translation)
        translation *= np.array(self.view_axis)
        
        # Apply transformation
        transformation = np.eye(4)
        transformation[:3, 3] = translation
        self.pcd.transform(transformation)
        self.vis.update_geometry(self.pcd)
        return True

    def rotate_around_axis(self, angle, action, mods):
        if action == 0:
            return False
        rotation_axis = self.view_axis
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array(rotation_axis) * angle)
        self.pcd.rotate(rotation_matrix, center=self.pcd.get_center())
        self.vis.update_geometry(self.pcd)
        return True

    def stretch_along_view(self, stretch_factor, action, mods):
        if action == 0:
            return False
        # Calculate stretch transformation matrix
        transformation = np.eye(4)
        transformation[:3, :3] = np.eye(3) + np.outer(self.view_axis, self.view_axis) * stretch_factor
        
        # Get the base point for stretching
        # if stretch_factor > 0:
        if sum(self.view_axis) > 0:
            base_point = np.min(np.asarray(self.pcd.points) @ np.outer(self.view_axis, self.view_axis), axis=0)
        else:
            base_point = np.max(np.asarray(self.pcd.points) @ np.outer(self.view_axis, self.view_axis), axis=0)
        # base_point *= np.abs(np.array(self.view_axis))
        # base_point *= self.view_axis

        # Apply stretch transformation
        self.pcd.points = o3d.utility.Vector3dVector(
            (np.asarray(self.pcd.points) - base_point) @ transformation[:3, :3] + base_point
        )
        self.vis.update_geometry(self.pcd)
        return True

    def translate_or_stretch(self, param1, param2, action, mods):
        if mods == 0:
            return self.translate_view_plane(param1, action, mods)
        elif mods == 1:
            return self.stretch_along_view(param2, action, mods)
        else:
            return False

    def run(self):
        self.vis.create_window()
        self.vis.add_geometry(self.axis)
        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(self.pc_non_edit)

        # Register key action callback for specific keys
        self.vis.register_key_action_callback(ord("X"), self.set_axis_view_x)
        self.vis.register_key_action_callback(ord("Y"), self.set_axis_view_y)
        self.vis.register_key_action_callback(ord("Z"), self.set_axis_view_z)

        # I and O keys for rotations
        self.vis.register_key_action_callback(ord("A"), lambda vis, action, mods: self.rotate_around_axis(-np.pi/36, action, mods))  # Rotate counter-clockwise
        self.vis.register_key_action_callback(ord("D"), lambda vis, action, mods: self.rotate_around_axis(np.pi/36, action, mods))  # Rotate clockwise

        # W, A, S, D keys for stretching
        self.vis.register_key_action_callback(ord("W"), lambda vis, action, mods: self.translate_or_stretch([0.01]*3, 0.1, action, mods))  # Stretch upwards
        self.vis.register_key_action_callback(ord("S"), lambda vis, action, mods: self.translate_or_stretch([-0.01]*3, 10./11.-1, action, mods))  # Stretch leftwards

        self.vis.run()
        self.vis.destroy_window()
    
    
if __name__ == '__main__':
    binaries = np.load('binaries.npy')
    grid_coords = np.load('grid_coords.npy')
    full_index = np.arange(len(binaries.reshape([-1])))
    inds = binaries.reshape([-1])
    data = grid_coords[inds]
    full_index = full_index[inds]
    color = np.zeros(data.shape)
    data = np.concatenate([data, color], axis=-1)
    post_str = ''
    
    # set total interested region num here
    part_num = 6

    for i in range(part_num):
        if i == part_num - 1:
            print("====save all remaining points====")
            np.save(f'occ_index_{i}_gs{post_str}.npy', full_index)
            break
        mask = np.zeros(len(data), dtype=bool)
        pcd_non_edit = o3d.geometry.PointCloud()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(data[:, 3:])
        # o3d.visualization.draw_geometries([pcd])
        picked_id = pick_points(pcd)
        print(len(picked_id))
        picked_id = [p.index for p in picked_id]
        # demo_manual_registration()
        mask[picked_id] = True
        if len(picked_id) == 0:
            picked_id = np.arange(len(data))
            mask = np.ones(len(data), dtype=bool)

        np.save(f'occ_mask_{i}_gs_{post_str}.npy', mask)
        np.save(f'occ_index_{i}_gs_{post_str}.npy', full_index[mask])
        print(f"===={i}-th occ mask sum index ====", mask.sum(), len(full_index[mask]))
        print("===mask saved===")
        data = data[~mask]
        full_index = full_index[~mask]