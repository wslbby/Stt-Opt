
from datasets import points_utils
from models import base_model
from models.backbone.pointnet import MiniPointNet, SegPointNet

import torch
from torch import nn
import torch.nn.functional as F

from utils.metrics import estimateOverlap, estimateAccuracy
from torchmetrics import Accuracy
import open3d as o3d
import numpy as np



class M2TRACK(base_model.MotionBaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.seg_acc = Accuracy(num_classes=2, average='none')
        self.last_loss = None
        self.box_aware = getattr(config, 'box_aware', False)
        self.use_motion_cls = getattr(config, 'use_motion_cls', False)
        # self.use_second_stage = getattr(config, 'use_second_stage', True)
        self.use_second_stage = False
        self.use_prev_refinement = getattr(config, 'use_prev_refinement', True)
        self.seg_pointnet = SegPointNet(input_channel=3 + 1 + 1 + (9 if self.box_aware else 0),
                                        per_point_mlp1=[64, 64, 64, 128, 1024],
                                        per_point_mlp2=[512, 256, 128, 128],
                                        output_size=2 + (9 if self.box_aware else 0))
        self.mini_pointnet = MiniPointNet(input_channel=3 + 1 + (9 if self.box_aware else 0),
                                          per_point_mlp=[64, 128, 256, 512],
                                          hidden_mlp=[512, 256],
                                          output_size=-1)
        if self.use_second_stage:
            self.mini_pointnet2 = MiniPointNet(input_channel=3 + (9 if self.box_aware else 0),
                                               per_point_mlp=[64, 128, 256, 512],
                                               hidden_mlp=[512, 256],
                                               output_size=-1)

            self.box_mlp = nn.Sequential(nn.Linear(256, 128),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(),
                                         nn.Linear(128, 128),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(),
                                         nn.Linear(128, 4))
        if self.use_prev_refinement:
            self.final_mlp = nn.Sequential(nn.Linear(256, 128),
                                           nn.BatchNorm1d(128),
                                           nn.ReLU(),
                                           nn.Linear(128, 128),
                                           nn.BatchNorm1d(128),
                                           nn.ReLU(),
                                           nn.Linear(128, 4))
        if self.use_motion_cls:
            self.motion_state_mlp = nn.Sequential(nn.Linear(256, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 2))
            self.motion_acc = Accuracy(num_classes=2, average='none')

        self.motion_mlp = nn.Sequential(nn.Linear(256, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, 4))
    def rotz_batch_tensor(self, angles):
        w = torch.cos(angles / 2)
        x = torch.zeros_like(angles)
        y = torch.zeros_like(angles)
        z = -torch.sin(angles / 2)

        # 初始化旋转矩阵的空数组
        rotation_matrices = torch.zeros((angles.size(0), 3, 3)).cuda()

        # 计算四元数对应的旋转矩阵
        rotation_matrices[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        rotation_matrices[:, 0, 1] = 2 * (x*y - w*z)
        rotation_matrices[:, 0, 2] = 2 * (x*z + w*y)
        
        rotation_matrices[:, 1, 0] = 2 * (x*y + w*z)
        rotation_matrices[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        rotation_matrices[:, 1, 2] = 2 * (y*z - w*x)
        
        rotation_matrices[:, 2, 0] = 2 * (x*z - w*y)
        rotation_matrices[:, 2, 1] = 2 * (y*z + w*x)
        rotation_matrices[:, 2, 2] = 1 - 2 * (x**2 + y**2)
        return rotation_matrices
    
    def roty_batch_tensor(self, t):
        input_shape = t.shape
        output = torch.zeros(tuple(list(input_shape) + [3, 3]), dtype=torch.float32, device=t.device)
        c = torch.cos(t)
        s = torch.sin(t)
        output[..., 0, 0] = c
        output[..., 0, 2] = s
        output[..., 1, 1] = 1
        output[..., 2, 0] = -s
        output[..., 2, 2] = c
        return output

    def forward(self, input_dict):
        """
        Args:
            input_dict: {
            "points": (B,N,3+1+1)
            "candidate_bc": (B,N,9)

        }

        Returns: B,4

        """
        output_dict = {}
        x = input_dict["points"].transpose(1, 2)
        if self.box_aware:
            candidate_bc = input_dict["candidate_bc"].transpose(1, 2)
            x = torch.cat([x, candidate_bc], dim=1)

        B, _, N = x.shape

        seg_out = self.seg_pointnet(x)
        seg_logits = seg_out[:, :2, :]  # B,2,N
        pred_cls = torch.argmax(seg_logits, dim=1, keepdim=True)  # B,1,N
        mask_points = x[:, :4, :] * pred_cls
        mask_xyz_t0 = mask_points[:, :3, :N // 2]  # B,3,N//2
        mask_xyz_t1 = mask_points[:, :3, N // 2:]
        if self.box_aware:
            pred_bc = seg_out[:, 2:, :]
            mask_pred_bc = pred_bc * pred_cls
            # mask_pred_bc_t0 = mask_pred_bc[:, :, :N // 2]  # B,9,N//2
            # mask_pred_bc_t1 = mask_pred_bc[:, :, N // 2:]
            mask_points = torch.cat([mask_points, mask_pred_bc], dim=1)
            output_dict['pred_bc'] = pred_bc.transpose(1, 2)

        point_feature = self.mini_pointnet(mask_points)

        # motion state prediction
        motion_pred = self.motion_mlp(point_feature)  # B,4
        if self.use_motion_cls:
            motion_state_logits = self.motion_state_mlp(point_feature)  # B,2
            motion_mask = torch.argmax(motion_state_logits, dim=1, keepdim=True)  # B,1
            motion_pred_masked = motion_pred * motion_mask
            output_dict['motion_cls'] = motion_state_logits
        else:
            motion_pred_masked = motion_pred
        # previous bbox refinement
        if self.use_prev_refinement:
            prev_boxes = self.final_mlp(point_feature)  # previous bb, B,4
            output_dict["estimation_boxes_prev"] = prev_boxes[:, :4]
        else:
            prev_boxes = torch.zeros_like(motion_pred)

        # 1st stage prediction
        aux_box = points_utils.get_offset_box_tensor(prev_boxes, motion_pred_masked)

        # 2nd stage refinement
        if self.use_second_stage:
            mask_xyz_t0_2_t1 = points_utils.get_offset_points_tensor(mask_xyz_t0.transpose(1, 2),
                                                                     prev_boxes[:, :4],
                                                                     motion_pred_masked).transpose(1, 2)  # B,3,N//2
            mask_xyz_t01 = torch.cat([mask_xyz_t0_2_t1, mask_xyz_t1], dim=-1)  # B,3,N

            # transform to the aux_box coordinate system
            mask_xyz_t01 = points_utils.remove_transform_points_tensor(mask_xyz_t01.transpose(1, 2),
                                                                       aux_box).transpose(1, 2)

            if self.box_aware:
                mask_xyz_t01 = torch.cat([mask_xyz_t01, mask_pred_bc], dim=1)
            output_offset = self.box_mlp(self.mini_pointnet2(mask_xyz_t01))  # B,4
            output = points_utils.get_offset_box_tensor(aux_box, output_offset)
            output_dict["estimation_boxes"] = output
        else:
            output_dict["estimation_boxes"] = aux_box
        output_dict.update({"seg_logits": seg_logits,
                            "motion_pred": motion_pred,
                            'aux_estimation_boxes': aux_box,
                            })

        return output_dict
    
    def compute_corners(self, center, angle, wlh, prev_corners=None):
        # print(center.shape, angle.shape, wlh.shape)
        w, l, h = wlh[:, 0], wlh[:, 1], wlh[:, 2]
        b = w.shape[0]
        x_corners = l[:,None] / 2 * torch.tensor([1,  1,  1,  1, -1, -1, -1, -1]).cuda()
        y_corners = w[:,None] / 2 * torch.tensor([1, -1, -1,  1,  1, -1, -1,  1]).cuda()
        z_corners = h[:,None] / 2 * torch.tensor([1,  1, -1, -1,  1,  1, -1, -1]).cuda()
        corners = torch.stack((x_corners, y_corners, z_corners), dim=1)
        rotmat = self.rotz_batch_tensor(angle)
        corners = torch.bmm(rotmat.transpose(1,2).double(), corners)
        corners = corners.transpose(1,2)
        
        corners += center[:, None, :]

        # if prev_corners is not None:
        #     prev_corners = prev_corners.transpose(1,2)
        #     # prev_corners += center[:, None, :]
        #     # prev_corners = prev_corners.transpose(1,2)
        #     from mayavi import mlab
        #     # print(prev_corners)
        #     print(corners[0], prev_corners[0])
        #     for i in range(corners.shape[1]):
        #         mlab.points3d(corners[0, i, 0].detach().cpu().numpy(), corners[0, i, 1].detach().cpu().numpy(), corners[0, i, 2].detach().cpu().numpy(), color=(0, 0, 0))
        #         mlab.points3d(prev_corners[0, i, 0].detach().cpu().numpy(), prev_corners[0, i, 1].detach().cpu().numpy(), prev_corners[0, i, 2].detach().cpu().numpy(), color=(1, 0, 0))
        #     mlab.show()
        
        return corners
    
    
    def compute_triangle(self, prev_corners, aux_corners, prev_center, aux_center, diag):
        prev_diag, aux_diag = prev_corners[:,diag[0],:], aux_corners[:,diag[1],:]
        o = (prev_diag + aux_diag) / 2
        # 三角形的两条边
        oo1 = prev_center - o
        oo2 = aux_center - o
        o1o2 = aux_center - prev_center

        centroid = (prev_center + aux_center + o) / 3
        d_oo1 = torch.norm(oo1, dim=1)
        d_oo2 = torch.norm(oo2, dim=1)
        d_o1o2 = torch.norm(o1o2, dim=1)

        # sum_of_sides = d_oo1 + d_oo2 + d_o1o2
        # # I = (aux_center * d_oo1 + prev_center * d_oo2 + o * d_o1o2) / sum_of_sides

        # I_x = (d_oo1 * aux_center[:, 0] + d_oo2 * prev_center[:, 0] + d_o1o2 * o[:, 0]) / sum_of_sides
        # I_y = (d_oo1 * aux_center[:, 1] + d_oo2 * prev_center[:, 1] + d_o1o2 * o[:, 1]) / sum_of_sides
        # I_z = (d_oo1 * aux_center[:, 2] + d_oo2 * prev_center[:, 2] + d_o1o2 * o[:, 2]) / sum_of_sides
        # I = torch.stack((I_x, I_y, I_z), dim=1)
        # 求空间三角形的法向量norm_vector
        # 求两边夹角也需要norm_vector
        norm_vector = torch.cross(oo1, oo2, dim=1)
        
        # print(norm_vector)
        norm_vector = F.normalize(norm_vector, p=2, dim=1)
        norm_vector = torch.cross(oo1, oo2, dim=1)


        # norm_vector = F.normalize(norm_vector, p=2, dim=1)   # 归一化法向量


        return {
            'o': o,
            'd_oo1': d_oo1,
            'd_oo2': d_oo2,
            'd_o1o2': d_o1o2,
            'oo1': oo1,
            'oo2': oo2,
            'norm_vec': norm_vector,
            'centroid': centroid
            # 'I': I
        }

    def compute_triangle_loss(self, pred_dict, gt_dict):

        loss_diag_center = F.smooth_l1_loss(pred_dict['o'], gt_dict['o'])
        loss_centroid =  F.smooth_l1_loss(pred_dict['centroid'], gt_dict['centroid'])
        # loss_I = F.smooth_l1_loss(pred_dict['I'], gt_dict['I'])

        length_loss = F.smooth_l1_loss(pred_dict['d_oo1'], gt_dict['d_oo1']) + \
                      F.smooth_l1_loss(pred_dict['d_oo2'], gt_dict['d_oo2']) + \
                      F.smooth_l1_loss(pred_dict['d_o1o2'], gt_dict['d_o1o2'])
                      
        
        # angle_pred = F.cosine_similarity(pred_dict['oo1'], pred_dict['oo2'], dim=1)
        # angle_gt = F.cosine_similarity(gt_dict['oo1'], gt_dict['oo2'], dim=1)
        # angle_loss = F.mse_loss(angle_pred, angle_gt)

        # print(pred_dict['oo1'].shape, pred_dict['norm_vec'].shape)
        # print(F.cosine_similarity(pred_dict['norm_vec'], gt_dict['norm_vec'], dim=1))

        norm_vector_loss = 1 - torch.mean(F.cosine_similarity(pred_dict['norm_vec'], gt_dict['norm_vec'], dim=1))
        # print(F.cosine_similarity(pred_dict['norm_vec'], gt_dict['norm_vec'], dim=1))
        

        return {'loss_diag_center': loss_diag_center,
                'length_loss': length_loss,
                # 'angle_loss': angle_loss,
                'norm_vector_loss': norm_vector_loss,
                'loss_centroid': loss_centroid}
                # 'loss_I': loss_I}

       

    def compute_loss(self, data, output):
        loss_total = 0.0
        loss_dict = {}
        aux_estimation_boxes = output['aux_estimation_boxes']  # B,4
        motion_pred = output['motion_pred']  # B,4
        seg_logits = output['seg_logits']
        with torch.no_grad():
            seg_label = data['seg_label']
            box_label = data['box_label']
            box_label_prev = data['box_label_prev']
            motion_label = data['motion_label']
            motion_state_label = data['motion_state_label']
            center_label = box_label[:, :3]
            angle_label = box_label[:, 3]
            center_label_prev = box_label_prev[:, :3]
            angle_label_prev = box_label_prev[:, 3]
            center_label_motion = motion_label[:, :3]
            angle_label_motion = torch.sin(motion_label[:, 3])
            this_corners = data['this_corners']
            prev_corners = data['prev_corners']
            bbox_size = data['bbox_size']

        loss_seg = F.cross_entropy(seg_logits, seg_label, weight=torch.tensor([0.5, 2.0]).cuda())
        # loss_center_motion = F.smooth_l1_loss(motion_pred[:, :3], center_label_motion)
        # loss_angle_motion = F.smooth_l1_loss(torch.sin(motion_pred[:, 3]), angle_label_motion)
        # loss_motion = loss_center_motion * self.config.center_weight + loss_angle_motion * self.config.angle_weight
        # loss_center_prev = F.smooth_l1_loss(estimation_boxes_prev[:, :3], center_label_prev)
        # loss_angle_prev = 10 * F.smooth_l1_loss(torch.sin(estimation_boxes_prev[:, 3]), angle_label_prev)
        # loss_prev = (loss_center_prev * self.config.center_weight + loss_angle_prev * self.config.angle_weight)
        # loss_dict["loss_center_prev"] = loss_center_prev
        # loss_dict["loss_angle_prev"] = loss_angle_prev

        # loss_center_aux = F.smooth_l1_loss(aux_estimation_boxes[:, :3], center_label)
        # loss_angle_aux = 10 * F.smooth_l1_loss(torch.sin(aux_estimation_boxes[:, 3]), angle_label)
        # loss_dict["loss_center_aux"] = loss_center_aux
        # loss_dict["loss_angle_aux"] = loss_angle_aux
        # loss_this = (loss_center_aux * self.config.center_weight + loss_angle_prev * self.config.angle_weight)

        loss_centroid, loss_I, loss_length, loss_diag_center, loss_normal_vector = 0.0, 0.0, 0.0, 0.0, 0.0

        wlh = bbox_size
        estimation_boxes_prev = output['estimation_boxes_prev']  # B,4
        estimation_prev_centers = estimation_boxes_prev[:, :3]
        estimation_prev_angles = estimation_boxes_prev[:, 3]
        estimation_prev_corners = self.compute_corners(estimation_prev_centers, estimation_prev_angles, wlh)

        aux_estimation_centers = aux_estimation_boxes[:, :3]
        aux_estimation_angles = aux_estimation_boxes[:, 3]
        aux_estimation_corners = self.compute_corners(aux_estimation_centers, aux_estimation_angles, wlh)

        angle_motion_gt = torch.sin(angle_label - angle_label_prev)
        angle_motion_pred = torch.sin(aux_estimation_angles - estimation_prev_angles)
        loss_angle_motion =  F.smooth_l1_loss(angle_motion_pred, angle_motion_gt)
        # todo 三点共线怎么办？


        for diag in [[1,7],[0,6],[2,4],[3,5]]:

            pred_dict = self.compute_triangle(estimation_prev_corners, aux_estimation_corners, estimation_prev_centers,
                                               aux_estimation_centers, diag=diag)
            
            gt_dict = self.compute_triangle(prev_corners.transpose(1,2), this_corners.transpose(1,2), 
                                            center_label_prev, center_label, diag=diag)
            
            loss_dict_triangle = self.compute_triangle_loss(pred_dict, gt_dict)
            loss_diag_center += loss_dict_triangle['loss_diag_center']
            loss_length += loss_dict_triangle['length_loss']
            loss_centroid += loss_dict_triangle['loss_centroid']
            # loss_angle += loss_dict_triangle['angle_loss']
            loss_normal_vector += loss_dict_triangle['norm_vector_loss']
            # loss_I += loss_dict_triangle['loss_I']

        loss_total = 10 * loss_seg +\
                              0.5 * loss_length +  loss_normal_vector + 10 * (loss_centroid) + 10 * loss_angle_motion
                              
        loss_dict.update({
            "loss_total": loss_total,
            "loss_seg": loss_seg,
            "loss_diag_center": loss_diag_center,
            "loss_centroid": loss_centroid,
            "loss_motion": 10 * loss_angle_motion,
            # "loss_I": loss_I,
            "loss_length": loss_length,
            # "loss_angle": loss_angle,
            "loss_normal_vector": loss_normal_vector,

        })
        if self.box_aware:
            prev_bc = data['prev_bc']
            this_bc = data['this_bc']
            bc_label = torch.cat([prev_bc, this_bc], dim=1)
            pred_bc = output['pred_bc']
            loss_bc = F.smooth_l1_loss(pred_bc, bc_label)
            loss_total += loss_bc * self.config.bc_weight
            loss_dict.update({
                "loss_total": loss_total,
                "loss_bc": loss_bc
            })

        return loss_dict
    
    def create_line_set(self, angle, center, size, color):

        angle = np.array([0, 0, angle + 1e-10])
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle(angle)
        bbox = o3d.geometry.OrientedBoundingBox(center, rot, size)

        corners = bbox.get_box_points()
        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
        # 设置颜色，例如红色
        

        lines = np.asarray(lineset.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

        lineset.lines = o3d.utility.Vector2iVector(lines)
        colors = np.array([color for i in range(len(lineset.lines))])  # 创建一个和线条数量相同的颜色数组
        lineset.colors = o3d.utility.Vector3dVector(colors)

        return lineset, corners
    
    def create_line(self, point1, point2, colors):
        
        lineset = o3d.geometry.LineSet()

        # point1, point2 = point1.reshape(-1,3), point2.reshape(-1,3)
        points = [point1, point2]
        lineset.points = o3d.utility.Vector3dVector(points)

        # 定义线段，即指定哪些顶点应该连接
        # 这里我们连接上面定义的两个点（点0和点1）
        lines = [[0, 1]]  # 线段由点集合中的索引定义
        lineset.lines = o3d.utility.Vector2iVector(lines)

        # 可以为线条设置颜色
        colors = [colors]  # 红色
        lineset.colors = o3d.utility.Vector3dVector(colors)

        return lineset


    def training_step(self, batch, batch_idx):
        """
        Args:
            batch: {
            "points": stack_frames, (B,N,3+9+1)
            "seg_label": stack_label,
            "box_label": np.append(this_gt_bb_transform.center, theta),
            "box_size": this_gt_bb_transform.wlh
        }
        Returns:

        """

        output = self(batch)
        loss_dict = self.compute_loss(batch, output)
        loss = loss_dict['loss_total']

        if self.last_loss is not None and loss - self.last_loss > 20:
            print(self.last_loss - loss)
            
                        # print(loss_dict)
            draw = 1
            if draw:
                pts = o3d.geometry.PointCloud()
                points = batch['points'][0, :, :3].reshape(-1,3).detach().cpu().numpy()
                print(batch['points'][0, :, :3].shape)
                pts.points = o3d.utility.Vector3dVector(points)

                prev_center = output['estimation_boxes_prev'][0, 0:3].detach().cpu().numpy().reshape(3,1)
                prev_angle = output['estimation_boxes_prev'][0, 3].detach().cpu().numpy()
                center = output['aux_estimation_boxes'][0, 0:3].detach().cpu().numpy().reshape(3,1)
                angle = output['aux_estimation_boxes'][0, 3].detach().cpu().numpy()

                print(center, prev_center)

                size = batch['bbox_size'][0].detach().cpu().numpy().reshape(3,1)
                size = [size[1], size[0], size[2]]

                prev_center_gt = batch['box_label_prev'][0, 0:3].detach().cpu().numpy().reshape(3,1)
                prev_angle_gt = batch['box_label_prev'][0, 3].detach().cpu().numpy()
                center_gt = batch['box_label'][0, 0:3].detach().cpu().numpy().reshape(3,1)
                angle_gt = batch['box_label'][0, 3].detach().cpu().numpy()

                # def create_line_set(self, angle, center, size, color):

                prev_bbox_gt, prev_bbox_gt_corners = self.create_line_set(prev_angle_gt, prev_center_gt, size, [1,0,0])
                prev_bbox, prev_bbox_corners = self.create_line_set(prev_angle, prev_center, size, [0,1,0])
                
                this_bbox_gt, this_bbox_gt_corners = self.create_line_set(angle_gt, center_gt, size, [1,0,0])
                this_bbox, this_bbox_corners = self.create_line_set(angle, center, size, [0,1,0])

                o1o2_gt = self.create_line(prev_center_gt, center_gt, colors=[1,0,0])
                o1o2 = self.create_line(prev_center, center, colors=[0,1,0])

                prev_bbox_gt_corners = np.asarray(prev_bbox_gt_corners)
                this_bbox_gt_corners = np.asarray(this_bbox_gt_corners)

                prev_bbox_corners = np.asarray(prev_bbox_corners)
                this_bbox_corners = np.asarray(this_bbox_corners)

                o_gt = (prev_bbox_gt_corners[4,:] + this_bbox_gt_corners[0,:]) / 2
                oo1_gt = self.create_line(prev_center_gt, o_gt, colors=[1,0,0])
                oo2_gt = self.create_line(center_gt, o_gt, colors=[1,0,0])
                diag_gt = self.create_line(prev_bbox_gt_corners[4,:], this_bbox_gt_corners[0,:], colors=[1,0,0])

                o = (prev_bbox_corners[4,:] + this_bbox_corners[0,:]) / 2
                oo1 = self.create_line(prev_center, o, colors=[0,1,0])
                oo2 = self.create_line(center, o, colors=[0,1,0])
                diag = self.create_line(prev_bbox_corners[4,:], this_bbox_corners[0,:], colors=[0,1,0])

                # # def compute_corners(self, center, angle, wlh):
                # prev_gt_corners = self.compute_corners(prev_center_gt.reshape(1,3), prev_angle_gt.reshape(1,1), np.array(size.reshape(1,3)))

                # prev_center_gt = self.create_points(prev_center_gt.reshape(-1,3), colors=[1,0,0])
                # prev_center = self.create_points(prev_center.reshape(-1,3), colors=[1,0,0])

                # this_center_gt = self.create_points(center_gt.reshape(-1,3), colors=[1,0,0])
                # this_center = self.create_points(center.reshape(-1,3), colors=[1,0,0])
                geos = [pts, prev_bbox_gt, prev_bbox, this_bbox_gt, this_bbox,
                                                o1o2_gt, oo1_gt, oo2_gt, diag_gt,
                                                o1o2, oo1, oo2, diag]
                o3d.visualization.draw_geometries(geos)
                # self.logger.experiment.add_3d('cube', to_dict_batch(geos), step=self.global_step)
                # self.logger.experiment.add_3d('geos', geos, step=self.global_step)
        self.last_loss = loss
        # log
        seg_acc = self.seg_acc(torch.argmax(output['seg_logits'], dim=1, keepdim=False), batch['seg_label'])
        self.log('seg_acc_background/train', seg_acc[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('seg_acc_foreground/train', seg_acc[1], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        if self.use_motion_cls:
            motion_acc = self.motion_acc(torch.argmax(output['motion_cls'], dim=1, keepdim=False),
                                         batch['motion_state_label'])
            self.log('motion_acc_static/train', motion_acc[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log('motion_acc_dynamic/train', motion_acc[1], on_step=True, on_epoch=True, prog_bar=False,
                     logger=True)

        # for k, v in loss_dict.items():
        #     print(k, v)
        log_dict = {k: v.item() for k, v in loss_dict.items()}
        


        self.logger.experiment.add_scalars('loss', log_dict,
                                           global_step=self.global_step)
        return loss
    
    def on_after_backward(self):
        # 每100个批次记录一次
        if self.global_step % 100 == 0:  # 注意 global_step 从 1 开始
            # 只有在训练阶段才记录梯度
            if self.trainer.state.fn == 'fit':
                # 遍历模型的所有参数
                for name, param in self.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # 使用 logger 记录梯度
                        self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step=self.global_step)
                        
        


