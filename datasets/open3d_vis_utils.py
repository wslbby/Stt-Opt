"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
#import open3d
import torch
import matplotlib
import numpy as np
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mayavi import mlab
import open3d as o3d
box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num + 1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None,
                draw_origin=True, filepath=None, new_points=None, new_points_from_file=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(new_points, torch.Tensor):
        new_points = new_points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    if isinstance(point_colors, torch.Tensor):
        point_colors = point_colors.detach().cpu().numpy()

    # vis = open3d.visualization.Visualizer()
    # vis.create_window()
    #
    # vis.get_render_option().point_size = 3.0
    # vis.get_render_option().background_color = np.ones(3)

    # draw origin
    # if draw_origin:
    #     axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    #     vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    # vis.add_geometry(pts)
    # new_color = np.array([[1, 0, 0]]).repeat(points.shape[0], axis=0).reshape(-1, 3)

    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.zeros((points.shape[0], 3)))
        # pts.colors = open3d.utility.Vector3dVector(new_color)
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    # if gt_boxes is not None:
    #     box = draw_box(gt_boxes, (0, 0, 1))
    #     open3d.visualization.draw_geometries([pts, box])
    # if ref_boxes is not None:
    #     box = draw_box(ref_boxes, (0, 1, 0), ref_labels, ref_scores)
    #     open3d.visualization.draw_geometries([pts, box])
    if new_points is None:
        if gt_boxes is not None:
            box = draw_box(gt_boxes, (0, 0, 1))
            open3d.visualization.draw_geometries([pts, box])
        else:
            open3d.visualization.draw_geometries([pts])
    if new_points is not None:
        if gt_boxes is not None:
            box = draw_box(gt_boxes, (0, 0, 1))
            open3d.visualization.draw_geometries([pts, box])
            new_pts = open3d.geometry.PointCloud()
            new_pts.points = open3d.utility.Vector3dVector(new_points[:, :3])
            new_color = np.array([[1, 0, 0]]).repeat(new_points.shape[0], axis=0).reshape(-1, 3)
            # print(new_color)
            new_pts.colors = open3d.utility.Vector3dVector(new_color)
            open3d.visualization.draw_geometries([pts, new_pts, box])
        else:
            new_pts = open3d.geometry.PointCloud()
            new_pts.points = open3d.utility.Vector3dVector(new_points[:, :3])
            new_color = np.array([[1, 0, 0]]).repeat(new_points.shape[0], axis=0).reshape(-1, 3)
            # print(new_color)
            new_pts.colors = open3d.utility.Vector3dVector(new_color)
            open3d.visualization.draw_geometries([pts, new_pts])
    if new_points_from_file is not None:
        new_points = np.loadtxt(new_points_from_file)
        new_pts = open3d.geometry.PointCloud()
        new_pts.points = open3d.utility.Vector3dVector(new_points[:, :3])
        new_color = np.array([[1, 0, 0]]).repeat(new_points.shape[0], axis=0).reshape(-1, 3)
        # print(new_color)
        new_pts.colors = open3d.utility.Vector3dVector(new_color)
        open3d.visualization.draw_geometries([pts, new_pts])


def save_points(points, filepath):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if not os.path.exists(filepath):
        os.system(r"touch {}".format(filepath))
    np.savetxt(filepath, points)


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    wlh = gt_boxes[3:6]
    lwh = [wlh[1], wlh[0], wlh[2]]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes)
    if ref_labels is None:
        line_set.paint_uniform_color(color)
    else:
        line_set.paint_uniform_color(box_colormap[ref_labels])

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return line_set


def draw_pttr(points, gt_boxes=None, my_boxes=None, pttr_boxes=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(pttr_boxes, torch.Tensor):
        pttr_boxes = pttr_boxes.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(my_boxes, torch.Tensor):
        my_boxes = my_boxes.cpu().numpy()

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    color = np.array([[0, 0, 0]]).repeat(points.shape[0], axis=0).reshape(-1, 3)
    # print(new_color)
    pts.colors = open3d.utility.Vector3dVector(color)

    gt_boxes = draw_box(gt_boxes, (0, 0, 1))
    pttr_boxes = draw_box(pttr_boxes, (0, 1, 0))
    my_boxes = draw_box(my_boxes, (1, 0, 0))
    open3d.visualization.draw_geometries([pts, pttr_boxes, my_boxes, gt_boxes])


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid[np.newaxis, :]
    m = np.max(np.sqrt(np.sum(np.power(pc, 2), axis=1)), axis=0)
    pc = pc / m[np.newaxis, np.newaxis]
    return pc

def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,0,0), line_width=4, center=None) -> object:


    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]

        for k in range(0,4):

            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+3)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    if center is not None:
        mlab.points3d(center)
    # mlab.view(azimuth=160, elevation=None, distance=20, focalpoint=[2.0909996, -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991
    # mlab.show()

def  plt_scenes(points, attn_score=None, labels=None, gt_boxes=None, point_colors=None, final_pts=None, index=None, new_points=None, flow=None, pred_boxes=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(final_pts, torch.Tensor):
        final_pts = final_pts.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(point_colors, torch.Tensor):
        point_colors = point_colors.detach().cpu().numpy()
    if isinstance(attn_score, torch.Tensor):
        attn_score = attn_score.detach().cpu().numpy()
    if isinstance(new_points, torch.Tensor):
        new_points = new_points.detach().cpu().numpy()
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()
    if isinstance(pred_boxes, torch.Tensor):
        pred_boxes = pred_boxes.detach().cpu().numpy()



    fig = mlab.figure(bgcolor=(1, 1, 1), size=(1200, 800))
    # mlab.plot3d()
    # mlab.points3d(points[:,0], points[:,1], points[:,2], point_colors, scale_mode='none', mode='point', colormap='rainbow',scale_factor=.03, figure=fig)

    if labels is not None:
        if point_colors is None:
            if labels.shape[0] == 1:
                labels = labels.squeeze(0)
            else:
                pass
            f_points = points[labels == 1]
            b_points = points[labels == 0]
            # mlab.points3d(f_points[:, 0], f_points[:, 1], f_points[:, 2], color=(1, 0, 0), mode='axes', scale_factor=0.05)
            # mlab.points3d(b_points[:, 0], b_points[:, 1], b_points[:, 2], color=(0, 0, 0), mode='2dcircle', scale_factor=0.03)
            mlab.points3d(f_points[:, 0], f_points[:, 1], f_points[:, 2], color=(1, 0, 0),
                          scale_factor=0.20)
            mlab.points3d(b_points[:, 0], b_points[:, 1], b_points[:, 2], color=(0, 0, 0),
                          scale_factor=0.10)
            if index is not None:
                mlab.savefig('/home/zhaokj_home/projects/preserve/trackid:{},index:{},recall:{:.2f}.png'.format(index[0], index[1], index[2]))
            else:
                mlab.show()
        elif point_colors is not None:
            print(1)
            # mlab.points3d(points[:, 0], points[:, 1], points[:, 2], point_colors, scale_mode='none', scale_factor=0.05,
            #               colormap='rainbow')
            labels = labels.squeeze(0)
            f_points = points[labels == 1]
            b_points = points[labels == 0]
            f_colors = point_colors[labels == 1]
            b_colors = point_colors[labels == 0]

            mlab.points3d(f_points[:, 0], f_points[:, 1], f_points[:, 2], f_colors, scale_mode='none', scale_factor=0.12,
                          colormap='rainbow')
            mlab.points3d(b_points[:, 0], b_points[:, 1], b_points[:, 2], b_colors, scale_mode='none', scale_factor=0.12,
                          colormap='rainbow')
            # mlab.points3d(final_pts[:, 0], final_pts[:, 1], final_pts[:, 2], color=(1,0,0), mode='axes',scale_mode='none',
            #               scale_factor=0.2)

    else:
        if new_points is not None:
            if flow is not None:

                print(1)
                mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color=(0, 0, 0), scale_factor=0.12)
                mlab.points3d(new_points[:, 0], new_points[:, 1], new_points[:, 2], color=(1, 0, 0), scale_factor=0.12)
                #print(flow)
                # flow[:,[1,2]] = flow[:,[2,1]]
                wrapped = points + flow
                if len(points.shape) == 3:
                    points = points.squeeze(0).T
                    wrapped = wrapped.squeeze(0).T
                # mlab.points3d(wrapped[:, 0], wrapped[:, 1], wrapped[:, 2], color=(0, 0, 1), scale_factor=0.12)
                # mlab.points3d(flow[0], flow[1], flow[2], color=(0, 0, 1), scale_factor=0.5)
                for i in range(128):
                    mlab.plot3d([points[i, 0], wrapped[i, 0]], [points[i, 1], wrapped[i, 1]], [points[i, 2], wrapped[i, 2]], color=(0, 1, 0))
            else:
                mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color=(0, 0, 0), scale_factor=0.12)
                mlab.points3d(new_points[:, 0], new_points[:, 1], new_points[:, 2], color=(1, 0, 0), scale_factor=0.12)
        else:
            mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color=(0, 0, 0), scale_factor=0.08)

    if gt_boxes is not None:
        draw_gt_boxes3d(gt_boxes, fig=fig, line_width=5)
    if pred_boxes is not None:
        draw_gt_boxes3d(pred_boxes, fig=fig, line_width=5, color=(0,0,1))
    # mlab.savefig('/home/zhaokj_home/projects/preserve/trackid:{}.png'.format(index))
    mlab.orientation_axes()
    # mlab.view(azimuth=120, elevation=None, distance=30, focalpoint=[2.0909996, -1.04700089, -2.03249991])
    mlab.show()
    mlab.close()

def mlab_pttr(search_area, my_box, pttr_box, gt_box, index):
    if isinstance(search_area, torch.Tensor):
        search_area = search_area.cpu().numpy()
    if isinstance(my_box, torch.Tensor):
        my_box = my_box.cpu().numpy()
    if isinstance(pttr_box, torch.Tensor):
        pttr_box = pttr_box.cpu().numpy()
    if isinstance(gt_box, torch.Tensor):
        gt_box = gt_box.cpu().numpy()
    fig = mlab.figure(bgcolor=(1, 1, 1), size=(1200, 800))

    mlab.points3d(search_area[:, 0], search_area[:, 1], search_area[:, 2], color=(0, 0, 0), scale_factor=0.08)
    draw_gt_boxes3d(my_box, fig=fig, line_width=5, color=(0,0,1))
    draw_gt_boxes3d(pttr_box, fig=fig, line_width=5, color=(0,1,0))
    draw_gt_boxes3d(gt_box, fig=fig, line_width=5,color=(1,0,0))
    mlab.savefig('/home/zhaokj_home/projects/result/%d.png' % (index))
    mlab.close()

def mlab_trace(pcs, stnets, m2tracks, cxtracks, rands, gts, draw_trace=False):
    # if isinstance(pred, torch.Tensor):
    #     pred = pred.cpu().numpy()
    if isinstance(gts, torch.Tensor):
        gts = gts.cpu().numpy()
    fig = mlab.figure(bgcolor=(1, 1, 1), size=(1200, 800))

    
    if draw_trace:
        for i, (pc, stnet, m2track, cxtrack, rand, gt_bbox) in enumerate(zip(pcs, stnets, m2tracks, cxtracks, rands, gts)):
            gap = 5
            m2track = m2track.center
            gt_bbox = gt_bbox.center
            rand = rand.center
            # rand = gt_bbox + np.random.uniform(low=-0.05, high=0.05, size=3)
            # print(points.shape)
            # fig = mlab.figure(bgcolor=(1, 1, 1), size=(1200, 800))
            if i % gap == 0:
                # print(stnet, m2track, gt_bbox)
                
                for model, color in zip([stnet, m2track, cxtrack, rand, gt_bbox],
                                                    [(1, 0, 0),(0, 1, 0), (0, 0, 0), (0, 1, 1), (0, 0, 1)]):
                    mlab.points3d(model[0], model[1], model[2], color=color, scale_factor=0.12)
                if i > 0:
                    for model, models, color in zip([stnet, m2track, cxtrack, rand, gt_bbox],
                                                    [stnets, m2tracks, cxtracks, rands, gts],
                                                    [(1, 0, 0),(0, 1, 0), (0, 0, 0), (0, 1, 1), (0, 0, 1)]):
                        if isinstance(models[0], np.ndarray):
                            mlab.plot3d([model[0], models[i-gap][0]], [model[1], models[i-gap][1]], [model[2], models[i-gap][2]], color=color)
                        else:
                            mlab.plot3d([model[0], models[i-gap].center[0]], [model[1], models[i-gap].center[1]], [model[2], models[i-gap].center[2]], color=color)
            if i == len(pcs)-1:
                for model, color in zip([stnet, m2track, cxtrack, rand, gt_bbox],
                                                    [(1, 0, 0),(0, 1, 0), (0, 0, 0), (0, 1, 1), (0, 0, 1)]):
                    mlab.points3d(model[0], model[1], model[2], color=color, scale_factor=0.12)
                for model, models, color in zip([stnet, m2track, cxtrack, rand, gt_bbox],
                                                    [stnets, m2tracks, cxtracks, rands, gts],
                                                    [(1, 0, 0),(0, 1, 0), (0, 0, 0), (0, 1, 1), (0, 0, 1)]):
                        if isinstance(models[0], np.ndarray):
                            mlab.plot3d([model[0], models[len(pcs)-1-(len(pcs)-1)%gap][0]], 
                                        [model[1], models[len(pcs)-1-(len(pcs)-1)%gap][1]], 
                                        [model[2], models[len(pcs)-1-(len(pcs)-1)%gap][2]], color=color)
                        else:
                            mlab.plot3d([model[0], models[len(pcs)-1-(len(pcs)-1)%gap].center[0]], 
                                        [model[1], models[len(pcs)-1-(len(pcs)-1)%gap].center[1]], 
                                        [model[2], models[len(pcs)-1-(len(pcs)-1)%gap].center[2]], color=color)
                
    mlab.show()
                
    # pcd_o3d = o3d.geometry.PointCloud()
    # pcd_o3d.points = o3d.utility.Vector3dVector(pcs[20].points.T)
    # _, inliers = pcd_o3d.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=200)
    # pcd_ = pcd_o3d.select_by_index(inliers, invert=True)
    # # print(np.asarray(pcd_.points).shape)
    # # from utils.open3d_vis_utils import plt_scenes
    # # plt_scenes(np.asarray(pcd_.points))
    # points = np.asarray(pcd_.points)
    
    if not draw_trace:
        for i in range(len(gts)):
            if i % 5 == 0:
                fig = mlab.figure(bgcolor=(1, 1, 1), size=(1200, 800))
                print(i)
                frame_id = i
                points = pcs[frame_id].points.T
                mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color=(0, 0, 0), scale_factor=0.05, figure=fig)
                # print(stnets[10][3:].shape)
                draw_gt_boxes3d(stnets[frame_id][3:].reshape(3,8).T.reshape(1,8,3), fig, color=(1, 0, 0))
                draw_gt_boxes3d(cxtracks[frame_id][3:].reshape(3,8).T.reshape(1,8,3), fig, color=(0, 0, 0))
                draw_gt_boxes3d(m2tracks[frame_id].corners().T.reshape(1,8,3), fig, color=(0, 1, 0))
                draw_gt_boxes3d(gts[frame_id].corners().T.reshape(1,8,3), fig, color=(0, 0, 1))
                draw_gt_boxes3d(rands[frame_id].corners().T.reshape(1,8,3), fig, color=(0, 1, 1))
                mlab.show()
            if i == len(gts)-1:
                fig = mlab.figure(bgcolor=(1, 1, 1), size=(1200, 800))
                print(i)
                frame_id = i
                points = pcs[frame_id].points.T
                mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color=(0, 0, 0), scale_factor=0.05, figure=fig)
                # print(stnets[10][3:].shape)
                draw_gt_boxes3d(stnets[frame_id][3:].reshape(3,8).T.reshape(1,8,3), fig, color=(1, 0, 0))
                draw_gt_boxes3d(cxtracks[frame_id][3:].reshape(3,8).T.reshape(1,8,3), fig, color=(0, 0, 0))
                draw_gt_boxes3d(m2tracks[frame_id].corners().T.reshape(1,8,3), fig, color=(0, 1, 0))
                draw_gt_boxes3d(gts[frame_id].corners().T.reshape(1,8,3), fig, color=(0, 0, 1))
                draw_gt_boxes3d(rands[frame_id].corners().T.reshape(1,8,3), fig, color=(0, 1, 1))
                mlab.show()
                
                    
        # if np.sqrt(np.sum((pred_bbox.center - gt_bbox.center) ** 2)) > 5:
        #     plt_scenes(x1.T, new_points=x2.T, gt_boxes=gt_bbox.corners().T.reshape(1,8,3))
        # print(pred_bbox.center)
        # print(gt_bbox.center)
    # mlab.show()
    # mlab.close()
    # mlab.points3d(search_area[:, 0], search_area[:, 1], search_area[:, 2], color=(0, 0, 0), scale_factor=0.08)