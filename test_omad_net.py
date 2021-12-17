import argparse
import numpy as np
import os.path as osp
import open3d as o3d
import tqdm
import copy
import time
import os
import psutil
import mmcv
import torch.utils.data
from matplotlib import cm
import torch.multiprocessing as mp
from scipy.spatial.transform import Rotation as R
from dataset.dataset_omadnet import SapienDataset_OMADNet
from model.omad_net import OMAD_Net
from libs.utils import iou_3d, get_part_bbox_from_kp, get_part_bbox_from_corners, calc_joint_errors

cate_list = ['laptop', 'eyeglasses', 'dishwasher', 'drawer', 'scissors']


def rot_diff_rad(rot1, rot2):
    if np.abs((np.trace(np.matmul(rot1, rot2.T)) - 1) / 2) > 1.:
        print('Something wrong in rotation error!')
    return np.arccos((np.trace(np.matmul(rot1, rot2.T)) - 1) / 2) % (2*np.pi)


def rot_diff_degree(rot1, rot2):
    return rot_diff_rad(rot1, rot2) / np.pi * 180


class PoseEstimator(torch.nn.Module):
    def __init__(self, model, num_parts, num_kp, init_base_r, init_base_t, init_joint_state,
                 init_beta, part_kp_weight, device, joint_type='revolute', reg_weight=0.0):
        super(PoseEstimator, self).__init__()
        self.model = model.to(device)
        self.model.eval()
        self.num_parts = num_parts
        self.num_kp = num_kp
        self.num_joints = num_parts - 1
        self.device = device
        self.part_kp_weight = part_kp_weight
        self.joint_type = joint_type
        assert joint_type in ('revolute', 'prismatic')
        self.reg_weight = reg_weight

        x, y, z, w = R.from_matrix(init_base_r.cpu().numpy()).as_quat()
        self.base_r_quat = torch.nn.Parameter(torch.tensor(
            [w, x, y, z], device=device, dtype=torch.float), requires_grad=True)  # q=a+bi+ci+di
        self.base_t = torch.nn.Parameter(init_base_t.clone().detach().to(device), requires_grad=True)
        self.joint_state = torch.nn.Parameter(init_joint_state.clone().detach().to(device), requires_grad=True)
        self.beta = torch.nn.Parameter(init_beta.clone().detach().to(device), requires_grad=True)

    def forward(self, pred_kp, mode='base'):
        assert mode in ('base', 'joint_single', 'all')
        norm_kp = self.model.get_norm_keypoints(self.beta)[0]  # bs=1
        homo_norm_kp = torch.cat([norm_kp, torch.ones(norm_kp.shape[0], norm_kp.shape[1], 1, device=norm_kp.device)], dim=-1)
        homo_pred_kp = torch.cat([pred_kp, torch.ones(pred_kp.shape[0], pred_kp.shape[1], 1, device=pred_kp.device)], dim=-1)

        base_r_quat = self.base_r_quat / torch.norm(self.base_r_quat)
        a, b, c, d = base_r_quat[0], base_r_quat[1], base_r_quat[2], base_r_quat[3]  # q=a+bi+ci+di
        base_rot_matrix = torch.stack([1 - 2 * c * c - 2 * d * d, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d,
                                       2 * b * c + 2 * a * d, 1 - 2 * b * b - 2 * d * d, 2 * c * d - 2 * a * b,
                                       2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d,
                                       1 - 2 * b * b - 2 * c * c]).reshape(3, 3)
        base_transform = torch.cat([torch.cat([base_rot_matrix, self.base_t.transpose(0, 1)], dim=1),
                                    torch.tensor([[0., 0., 0., 1.]], device=self.device)], dim=0)
        base_objective = torch.mean(
            torch.norm(base_transform.matmul(homo_norm_kp[0].T).T - homo_pred_kp[0], dim=-1) * self.part_kp_weight[0])
        all_objective = base_objective

        norm_joint_loc_all, norm_joint_axis_all = self.model.get_norm_joint_params(self.beta)
        norm_joint_loc_all, norm_joint_axis_all = norm_joint_loc_all[0], norm_joint_axis_all[0]  # bs=1
        norm_joint_axis_all = norm_joint_axis_all / torch.norm(norm_joint_axis_all, dim=-1, keepdim=True)
        norm_joint_params_all = (norm_joint_loc_all.detach(), norm_joint_axis_all.detach())
        new_joint_anchor_list = []
        new_joint_axis_list = []
        relative_transform_list = []
        for joint_idx in range(self.num_joints):
            part_idx = joint_idx + 1
            # TODO: support kinematic tree depth > 2
            norm_joint_loc, norm_joint_axis = norm_joint_loc_all[joint_idx], norm_joint_axis_all[joint_idx]  # bs=1
            homo_joint_anchor = torch.cat([norm_joint_loc, torch.ones(1, device=self.device)]).unsqueeze(1)
            new_joint_anchor = base_transform.matmul(homo_joint_anchor)[:3, 0]
            new_joint_axis = base_rot_matrix.matmul(norm_joint_axis)
            a, b, c = new_joint_anchor[0], new_joint_anchor[1], new_joint_anchor[2]
            u, v, w = new_joint_axis[0], new_joint_axis[1], new_joint_axis[2]
            if self.joint_type == 'revolute':
                cos = torch.cos(-self.joint_state[joint_idx])
                sin = torch.sin(-self.joint_state[joint_idx])
                relative_transform = torch.cat([torch.stack([u*u+(v*v+w*w)*cos, u*v*(1-cos)-w*sin, u*w*(1-cos)+v*sin,
                                                   (a*(v*v+w*w)-u*(b*v+c*w))*(1-cos)+(b*w-c*v)*sin,
                                                   u*v*(1-cos)+w*sin, v*v+(u*u+w*w)*cos, v*w*(1-cos)-u*sin,
                                                   (b*(u*u+w*w)-v*(a*u+c*w))*(1-cos)+(c*u-a*w)*sin,
                                                   u*w*(1-cos)-v*sin, v*w*(1-cos)+u*sin, w*w+(u*u+v*v)*cos,
                                                   (c*(u*u+v*v)-w*(a*u+b*v))*(1-cos)+(a*v-b*u)*sin]).reshape(3, 4),
                                                   torch.tensor([[0., 0., 0., 1.]], device=self.device)], dim=0)
            elif self.joint_type == 'prismatic':
                relative_transform = torch.cat([torch.cat([torch.eye(3, device=self.device),
                                                             (new_joint_axis*self.joint_state[joint_idx]).unsqueeze(1)], dim=1),
                                                torch.tensor([[0., 0., 0., 1.]], device=self.device)], dim=0)
            relative_transform_list.append(relative_transform.detach())
            child_objective = torch.mean(torch.norm(relative_transform.matmul(base_transform).matmul(homo_norm_kp[part_idx].T).T - homo_pred_kp[part_idx],
                        dim=-1) * self.part_kp_weight[part_idx])
            all_objective += child_objective

            new_joint_anchor_list.append(new_joint_anchor.detach())
            new_joint_axis_list.append(new_joint_axis.detach())
        all_objective /= self.num_parts
        all_objective += (self.beta * self.beta).mean() * self.reg_weight  # regularization loss
        new_joint_params_all = (torch.stack(new_joint_anchor_list, dim=0), torch.stack(new_joint_axis_list, dim=0))
        relative_transform_all = torch.stack(relative_transform_list, dim=0)
        return all_objective, base_transform.detach(), relative_transform_all, \
               new_joint_params_all, norm_joint_params_all, norm_kp.detach()


def optimize_pose(estimator, pred_kp, rank=0, use_initial=False):
    estimator.base_r_quat.requires_grad_(True)
    estimator.base_t.requires_grad_(True)
    estimator.joint_state.requires_grad_(True)
    estimator.beta.requires_grad_(True)
    if use_initial:
        pass
    else:
        optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-2)
        last_loss = 0.
        for iter in range(3000):  # base transformation(r+t) + beta + joint state
            loss, _, _, _, _, _ = estimator(pred_kp.detach(), mode='all')
            if iter % 50 == 0:
                if rank == 0:
                    print('base_r + base_t + joint state + beta: iter {}, loss={:05f}'.format(iter, loss.item()))
                if abs(last_loss - loss.item()) < 0.5*1e-3:
                    break
                last_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    _, base_transform, relative_transform_all, new_joint_params_all, norm_joint_params_all, norm_kp = estimator(pred_kp.detach())
    joint_state = estimator.joint_state.detach()
    beta = estimator.beta.detach()
    return base_transform, relative_transform_all, new_joint_params_all, norm_joint_params_all, norm_kp, joint_state, beta


def get_new_part_kp(cloud, pred_trans_part_kp, thr=0.1, use_raw_kp=False):
    bs, n = cloud.size(0), cloud.size(1)
    num_parts, num_kp_per_part = pred_trans_part_kp.size(1), pred_trans_part_kp.size(2)
    num_kp = num_parts * num_kp_per_part

    pred_trans_kp = pred_trans_part_kp.detach().reshape(bs, 1, -1, 3)
    cloud_expand = cloud.detach().reshape(bs, n, 1, 3).expand(-1, -1, num_kp, -1)  # (bs, n, m, 3)
    pred_trans_kp_expand = pred_trans_kp.expand(-1, n, -1, -1)  # (bs, n, m, 3)

    dist_square = torch.sum((cloud_expand - pred_trans_kp_expand) ** 2, dim=-1)  # (bs, n, m)
    min_dist_square, cloud_idxs = torch.min(dist_square, dim=1)  # (bs, m)

    kp_weight = torch.sign(thr ** 2 - min_dist_square) * 0.5 + 0.5  # (bs, m)
    part_kp_weight = kp_weight.reshape(bs, num_parts, -1)  # (bs, k, m/k)

    if use_raw_kp:
        new_part_kp = pred_trans_part_kp
    else:
        cloud_idxs_expand = cloud_idxs.unsqueeze(-1).expand(-1, -1, 3)  # (bs, m, 3)
        new_kp = torch.gather(cloud.detach(), dim=1, index=cloud_idxs_expand)  # (bs, m, 3)
        new_part_kp = new_kp.reshape(bs, num_parts, -1, 3)  # (bs, k, m/k, 3)

    return new_part_kp, part_kp_weight


def parallel_eval(pid, rank, results, model, opt, device):
    print()
    print('rank {} loading model sucessfully!'.format(rank))
    print()

    all_eval_results = dict()

    pps = psutil.Process(pid=pid)
    if rank == 0:
        results = tqdm.tqdm(results)
    for result in results:
        try:
            if pps.status() in (psutil.STATUS_DEAD, psutil.STATUS_STOPPED):
                print('Parent Process {} has stopped, rank {} quit now!!'.format(pid, rank))
                os._exit(0)

            sample_id = result['sample_id']
            pred_cls = torch.from_numpy(result['pred_cls']).to(device)
            pred_trans_part_kp = torch.from_numpy(result['pred_trans_part_kp']).to(device)
            pred_base_r = torch.from_numpy(result['pred_base_r']).to(device)
            pred_base_t = torch.from_numpy(result['pred_base_t']).to(device)
            pred_joint_state = torch.from_numpy(result['pred_joint_state']).to(device)
            pred_beta = torch.from_numpy(result['pred_beta']).to(device)
            pred_norm_part_kp = torch.from_numpy(result['pred_norm_part_kp']).to(device)
            cloud = torch.from_numpy(result['cloud']).to(device)
            gt_part_r = torch.from_numpy(result['gt_part_r']).to(device)
            gt_part_t = torch.from_numpy(result['gt_part_t']).to(device)
            gt_norm_joint_loc = torch.from_numpy(result['gt_norm_joint_loc']).to(device)
            gt_norm_joint_axis = torch.from_numpy(result['gt_norm_joint_axis']).to(device)
            gt_joint_state = torch.from_numpy(result['gt_joint_state']).to(device)[1:]  # ignore base joint
            gt_norm_part_kp = torch.from_numpy(result['gt_norm_part_kp']).to(device)
            gt_norm_part_corners = torch.from_numpy(result['gt_norm_part_corners']).to(device)
            cate = torch.from_numpy(result['cate']).to(device)
            urdf_id = torch.from_numpy(result['urdf_id']).to(device)

            new_pred_trans_part_kp, part_kp_weight = get_new_part_kp(cloud.unsqueeze(0), pred_trans_part_kp.unsqueeze(0),
                                                                     thr=opt.kp_thr, use_raw_kp=opt.use_raw_kp)
            init_part_kp_weight = part_kp_weight[0]
            new_pred_trans_part_kp = new_pred_trans_part_kp[0]
            if rank == 0:
                print()
                print('URDF id={}'.format(urdf_id))
                print('{} valid keypoints!'.format(torch.sum(init_part_kp_weight).to(torch.int).item()))

            init_base_t = pred_base_t.unsqueeze(0)
            init_base_r = pred_base_r

            gt_trans_joint_anchor = []
            gt_trans_joint_axis = []
            gt_base_transform = torch.cat([torch.cat([gt_part_r[0, :, :], gt_part_t[0, :, :].transpose(0, 1)], dim=1),
                                    torch.tensor([[0., 0., 0., 1.]], device=device)], dim=0)
            for joint_idx in range(opt.num_parts - 1):
                homo_joint_anchor = torch.cat([gt_norm_joint_loc[joint_idx],
                                               torch.ones(1, device=device)]).unsqueeze(1)
                gt_trans_joint_anchor.append(gt_base_transform.matmul(homo_joint_anchor)[:3, 0])
                gt_trans_joint_axis.append(gt_base_transform[:3, :3].matmul(gt_norm_joint_axis[joint_idx]))
            gt_trans_joint_anchor = torch.stack(gt_trans_joint_anchor, dim=0)
            gt_trans_joint_axis = torch.stack(gt_trans_joint_axis, dim=0)

            gt_r_list = []
            gt_t_list = []
            for part_idx in range(opt.num_parts):
                gt_r = gt_part_r[part_idx, :, :].cpu().numpy()
                gt_r_list.append(gt_r)
                gt_t = gt_part_t[part_idx, 0, :].cpu().numpy()
                gt_t_list.append(gt_t)
            gt_part_bbox = get_part_bbox_from_corners(gt_norm_part_corners)

            # TODO: support prismatic joints and kinematic tree depth > 2
            num_joints = opt.num_parts - 1
            init_joint_state = pred_joint_state
            joint_type = 'revolute' if opt.category != 4 else 'prismatic'
            if rank == 0:
                if joint_type == 'revolute':
                    print('init_joint_state={} degree'.format((init_joint_state.cpu().numpy() / np.pi * 180).tolist()))
                else:
                    print('init_joint_state={}'.format((init_joint_state.cpu().numpy()).tolist()))
            init_beta = pred_beta.unsqueeze(0)

            pose_estimator = PoseEstimator(model, opt.num_parts, opt.num_kp, init_base_r, init_base_t,
                                           init_joint_state, init_beta, init_part_kp_weight, device=device,
                                           joint_type=joint_type, reg_weight=opt.reg_weight)
            # ground-truth part-keypoint
            single_gt_trans_part_kp = torch.stack([gt_part_r[i, :, :].matmul(gt_norm_part_kp[i, :, :].T).T +
                                                   gt_part_t[i, :, :] for i in range(opt.num_parts)], dim=0)

            if opt.use_gt_kp:
                base_transform, relative_transform_all, pred_trans_joint_params_all, pred_norm_joint_params_all, \
                new_pred_norm_kp, new_pred_joint_state, new_pred_beta = optimize_pose(pose_estimator,
                                                                                      single_gt_trans_part_kp, rank=rank,
                                                                                      use_initial=opt.use_initial)
            else:
                base_transform, relative_transform_all, pred_trans_joint_params_all, pred_norm_joint_params_all, \
                    new_pred_norm_kp, new_pred_joint_state, new_pred_beta = optimize_pose(pose_estimator, new_pred_trans_part_kp,
                                                                                          rank=rank,
                                                                                          use_initial=opt.use_initial)

            pred_r_list = [base_transform[:3, :3].cpu().numpy()] + [
                           relative_transform_all[joint_idx].matmul(base_transform)[:3, :3].cpu().numpy()
                           for joint_idx in range(num_joints)]
            pred_t_list = [base_transform[:3, -1].cpu().numpy()] + [
                           relative_transform_all[joint_idx].matmul(base_transform)[:3, -1].cpu().numpy()
                           for joint_idx in range(num_joints)]
            pred_part_bbox = get_part_bbox_from_kp(new_pred_norm_kp)
            pred_norm_joint_loc, pred_norm_joint_axis = pred_norm_joint_params_all
            pred_trans_joint_loc, pred_trans_joint_axis = pred_trans_joint_params_all

            r_errs = []
            t_errs = []
            ious = []

            for part_idx in range(opt.num_parts):
                r_err = rot_diff_degree(pred_r_list[part_idx], gt_r_list[part_idx])
                t_err = np.linalg.norm(pred_t_list[part_idx] - gt_t_list[part_idx], axis=-1)
                iou = iou_3d(pred_part_bbox[part_idx], gt_part_bbox[part_idx])
                if rank == 0:
                    print('sample {}, urdf_id {}, part {}, r_error={:04f}, t_error={:04f}, iou_3d={:04f}'.format(
                        sample_id, urdf_id, part_idx, r_err, t_err, iou))
                r_errs.append(r_err)
                t_errs.append(t_err)
                ious.append(iou)
            joint_loc_errs = []
            joint_axis_errs = []
            joint_state_errs = []
            for joint_idx in range(num_joints):
                joint_loc_err, joint_axis_err, joint_state_err = calc_joint_errors(
                    pred_trans_joint_loc[joint_idx], pred_trans_joint_axis[joint_idx],
                    gt_trans_joint_anchor[joint_idx], gt_trans_joint_axis[joint_idx],
                    new_pred_joint_state[joint_idx], gt_joint_state[joint_idx], joint_type=joint_type)
                if rank == 0:
                    print('sample {}, urdf_id {}, joint {}, (camera space) '
                          'joint_loc_err={:4f}, joint_axis_err={:4f} degree, '
                          'joint_state_err={:4f}'.format(
                          sample_id, urdf_id, joint_idx, joint_loc_err, joint_axis_err, joint_state_err))
                joint_loc_errs.append(joint_loc_err)
                joint_axis_errs.append(joint_axis_err)
                joint_state_errs.append(joint_state_err)

            if rank == 0 and opt.show:
                base_colors = np.array([(207/255, 37/255, 38/255), (28/255, 108/255, 171/255),
                                        (38/255, 148/255, 36/255), (254/255, 114/255, 16/255)] * 2)
                cmap = cm.get_cmap("jet", opt.num_kp)
                kp_colors = cmap(np.linspace(0, 1, opt.num_kp, endpoint=True))[:, :3]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(cloud.cpu().numpy())
                pcd.colors = o3d.utility.Vector3dVector(base_colors[pred_cls.numpy(), :])

                pred_trans_static_kp_pcd_list = []
                pred_kp_pcd_list = []
                gt_kp_pcd_list = []
                pred_trans_norm_kp_pcd_list = []
                gt_trans_norm_kp_pcd_list = []
                pred_trans_norm_kp_mesh_list = []

                for part_idx in range(opt.num_parts):
                    pred_trans_static_kp = (pred_r_list[part_idx] @ gt_norm_part_kp[part_idx, :, :].cpu().numpy().T +
                                            pred_t_list[part_idx][np.newaxis, :].T).T
                    pred_trans_static_kp_pcd = o3d.geometry.PointCloud()
                    pred_trans_static_kp_pcd.points = o3d.utility.Vector3dVector(pred_trans_static_kp)
                    pred_trans_static_kp_pcd.paint_uniform_color([0, 0, 1])
                    pred_trans_static_kp_pcd_list.append(pred_trans_static_kp_pcd)

                    pred_kp_pcd = o3d.geometry.PointCloud()
                    pred_kp_pcd.points = o3d.utility.Vector3dVector(
                        new_pred_trans_part_kp[part_idx, :, :].cpu().numpy())
                    pred_kp_pcd.paint_uniform_color([0, 0, 0])
                    pred_kp_pcd_list.append(pred_kp_pcd)

                    gt_kp_pcd = o3d.geometry.PointCloud()
                    gt_kp_pcd.points = o3d.utility.Vector3dVector(
                        single_gt_trans_part_kp[part_idx, :, :].cpu().numpy())
                    gt_kp_pcd.paint_uniform_color([0, 0, 0])
                    gt_kp_pcd_list.append(gt_kp_pcd)

                    # optimized beta + pred r,t
                    pred_trans_norm_kp_pcd = o3d.geometry.PointCloud()
                    pred_trans_norm_kp = (pred_r_list[part_idx] @ new_pred_norm_kp[part_idx, :, :].cpu().numpy().T +
                                          pred_t_list[part_idx][np.newaxis, :].T).T
                    pred_trans_norm_kp_mesh_list += \
                        [o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=5).translate((x, y, z)) for
                        x, y, z
                        in pred_trans_norm_kp]
                    pred_trans_norm_kp_pcd.points = o3d.utility.Vector3dVector(pred_trans_norm_kp)
                    pred_trans_norm_kp_pcd.paint_uniform_color([1, 0, 1])
                    pred_trans_norm_kp_pcd_list.append(pred_trans_norm_kp_pcd)

                    # raw predicted beta + gt_r,t
                    gt_trans_norm_kp_pcd = o3d.geometry.PointCloud()
                    gt_trans_norm_kp = (gt_r_list[part_idx] @ pred_norm_part_kp[part_idx, :, :].cpu().numpy().T +
                                        gt_t_list[part_idx][np.newaxis, :].T).T
                    gt_trans_norm_kp_pcd.points = o3d.utility.Vector3dVector(gt_trans_norm_kp)
                    gt_trans_norm_kp_pcd.paint_uniform_color([1, 0, 1])
                    gt_trans_norm_kp_pcd_list.append(gt_trans_norm_kp_pcd)

                sphere_pts_num = np.asarray(pred_trans_norm_kp_mesh_list[0].vertices).shape[0]
                for idx, mesh in enumerate(pred_trans_norm_kp_mesh_list):
                    mesh.vertex_colors = o3d.utility.Vector3dVector(
                        kp_colors[np.newaxis, idx, :].repeat(sphere_pts_num, axis=0))

                line_pcd_list = []
                for joint_idx in range(num_joints):
                    start_point = pred_trans_joint_params_all[0][joint_idx].cpu().numpy()
                    end_point = start_point + pred_trans_joint_params_all[1][joint_idx].cpu().numpy()
                    line_points = np.stack([start_point, end_point])
                    lines = [[0, 1]]  # Right leg
                    colors = [[0, 0, 1] for i in range(len(lines))]
                    line_pcd = o3d.geometry.LineSet()
                    line_pcd.lines = o3d.utility.Vector2iVector(lines)
                    line_pcd.colors = o3d.utility.Vector3dVector(colors)
                    line_pcd.points = o3d.utility.Vector3dVector(line_points)
                    line_pcd_list.append(line_pcd)

                # o3d.visualization.draw_geometries(pred_kp_pcd_list + [pcd] + line_pcd_list
                #                                   + pred_trans_norm_kp_pcd_list)
                o3d.visualization.draw_geometries([pcd] + line_pcd_list + pred_trans_norm_kp_mesh_list)
            result_dict = dict()
            result_dict['r_err'] = r_errs
            result_dict['t_err'] = t_errs
            result_dict['iou_3d'] = ious
            result_dict['joint_loc_err'] = joint_loc_errs
            result_dict['joint_axis_err'] = joint_axis_errs
            result_dict['joint_state_err'] = joint_state_errs
            result_dict['beta'] = new_pred_beta.cpu().numpy()[0]
            result_dict['base_transform'] = base_transform.cpu().numpy()
            result_dict['relative_transform'] = relative_transform_all.cpu().numpy()
            result_dict['pred_r'] = pred_r_list
            result_dict['pred_t'] = pred_t_list
            result_dict['joint_params'] = pred_trans_joint_params_all[0].cpu().numpy(), pred_trans_joint_params_all[1].cpu().numpy()
            result_dict['joint_state'] = new_pred_joint_state.cpu().numpy()
            result_dict['norm_kp'] = new_pred_norm_kp.cpu().numpy()
            all_eval_results[sample_id] = result_dict
        except psutil.NoSuchProcess:
            print('Parent Process {} does not exist, rank {} quit now!!'.format(pid, rank))
            os._exit(0)
        except Exception as e:
            print(e)
            raise e
        if rank == 0:
            print()

    file_dir = osp.join(opt.work_dir, 'tmp_results')
    file_name = osp.join(opt.work_dir, 'tmp_results', '{}process_final_result_{}_{}.pkl'.format(
            opt.num_process, opt.data_postfix, rank))
    if not osp.exists(file_dir):
        os.mkdir(file_dir)
    mmcv.dump(all_eval_results, file_name)
    print('Rank {} finish writing {}!'.format(rank, file_name))


def collect_results(opt):
    final_results = dict()
    print('Collecting results!')
    for rank in range(opt.num_process):
        tmp_filename = osp.join(opt.work_dir, 'tmp_results', '{}process_final_result_{}_{}.pkl'.format(
            opt.num_process, opt.data_postfix, rank))
        while not osp.exists(tmp_filename):
            print('Didn''t find {}, wait for 10 seconds.....'.format(tmp_filename))
            time.sleep(10)
        tmp_results = mmcv.load(tmp_filename)
        final_results.update(tmp_results)
        del tmp_results
    return final_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='data', help='dataset root dir')
    parser.add_argument('--params_dir', type=str, help='the dir for params and kp annotations')
    parser.add_argument('--checkpoint', type=str, default=None, help='test checkpoint')
    parser.add_argument('--category', type=int, default=1, help='category to train')
    parser.add_argument('--num_points', type=int, default=1024, help='points')
    parser.add_argument('--num_kp', type=int, default=12, help='number of all keypoints')
    parser.add_argument('--num_cates', type=int, default=5, help='number of categories')
    parser.add_argument('--num_parts', type=int, default=2, help='number of parts')
    parser.add_argument('--num_basis', type=int, default=10, help='number of shape basis')
    parser.add_argument('--symtype', type=str, default='shape', choices=['shape', 'none'], help='the symmetry type')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--dense_soft_factor', type=float, default=1.0, help='the factor of dense softmax')
    parser.add_argument('--kp_thr', type=float, default=0.1, help='the threshold for kp weight')
    parser.add_argument('--reg_weight', type=float, default=0, help='the weight of regularization loss')
    parser.add_argument('--work_dir', type=str, default='work_dir/base', help='working dir')
    parser.add_argument('--out', type=str, required=True, help='output raw_results filename')
    parser.add_argument('--use_cache', action='store_true', help='whether to use cached tmp results')
    parser.add_argument('--num_process', type=int, default=8, help='the number of parallel processes')
    parser.add_argument('--data_postfix', type=str, default='v1', help='the postfix of .pkl')
    parser.add_argument('--show', action='store_true', help='wheter to visualize keypoints')
    parser.add_argument('--use_raw_kp', action='store_true', help='wheter to use raw keypoints')
    parser.add_argument('--use_gpu', action='store_true', help='wheter to use GPU')
    parser.add_argument('--use_gt_kp', action='store_true', help='wheter to use gt kp')
    parser.add_argument('--use_initial', action='store_true', help='wheter to use initial prediction without refinement')
    parser.add_argument('--shuffle', action='store_true', help='wheter to shuffle raw results')
    opt = parser.parse_args()

    device = torch.device("cuda:0")
    output_path = osp.join(opt.work_dir, opt.out)
    if osp.exists(output_path):
        results = mmcv.load(output_path)
    else:
        params_dict = torch.load(osp.join(opt.params_dir, 'params.pth'))
        model = OMAD_Net(device=device, params_dict=params_dict,
                           num_points=opt.num_points, num_kp=opt.num_kp, num_parts=opt.num_parts,
                           init_dense_soft_factor=opt.dense_soft_factor, num_basis=opt.num_basis, symtype=opt.symtype)
        model = model.to(device)

        assert opt.checkpoint is not None
        model.load_state_dict(torch.load(osp.join(opt.work_dir, opt.checkpoint), map_location=device))

        test_kp_anno_path = osp.join(opt.params_dir, 'unsup_test_keypoints.pkl')
        test_dataset = SapienDataset_OMADNet('val', data_root=opt.dataset_root, add_noise=False, num_pts=opt.num_points,
                                             num_parts=opt.num_parts, num_cates=opt.num_cates, cate_id=opt.category,
                                             device=torch.device('cpu'), kp_anno_path=test_kp_anno_path)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.bs, shuffle=False,
                                                      num_workers=opt.workers)

        assert opt.num_kp % opt.num_parts == 0, 'number of keypoints must be divided by number of parts'
        if not opt.show:
            test_dataloader = tqdm.tqdm(test_dataloader)

        model.eval()
        results = []
        sample_id = 0
        for j, data in enumerate(test_dataloader):
            cloud, gt_part_cls, gt_part_r, gt_part_quat, gt_part_t, gt_joint_state, gt_norm_joint_loc, gt_norm_joint_axis, \
            gt_norm_part_kp, gt_scale, gt_center, gt_norm_part_corners, cate, urdf_id = data

            cloud, gt_part_cls, gt_part_r, gt_part_quat, gt_part_t, gt_joint_state, \
            gt_norm_joint_loc, gt_norm_joint_axis, gt_norm_part_kp, gt_center = \
                cloud.to(device), gt_part_cls.to(device), gt_part_r.to(device), gt_part_quat.to(device), \
                gt_part_t.to(device), gt_joint_state.to(device), gt_norm_joint_loc.to(device), \
                gt_norm_joint_axis.to(device), gt_norm_part_kp.to(device), gt_center.to(device)

            with torch.no_grad():
                dense_part_cls_score, pred_trans_part_kp, pred_base_quat, pred_base_r, pred_base_t, pred_joint_state, \
                pred_beta, pred_norm_part_kp, pred_joint_loc, pred_joint_axis = model(cloud, None)
                pred_cls = torch.argmax(dense_part_cls_score, dim=-1)

            bs_now = cloud.size(0)
            for i in range(bs_now):
                result = dict(
                              sample_id=sample_id,
                              pred_cls=pred_cls[i].cpu().numpy(),
                              pred_trans_part_kp=pred_trans_part_kp[i].cpu().numpy(),
                              pred_base_quat=pred_base_quat[i].cpu().numpy(),
                              pred_base_r=pred_base_r[i].cpu().numpy(),
                              pred_base_t=pred_base_t[i].cpu().numpy(),
                              pred_joint_state=pred_joint_state[i].cpu().numpy(),
                              pred_beta=pred_beta[i].cpu().numpy(),
                              pred_norm_part_kp=pred_norm_part_kp[i].cpu().numpy(),
                              pred_joint_loc=pred_joint_loc[i].cpu().numpy(),
                              pred_joint_axis=pred_joint_axis[i].cpu().numpy(),
                              cloud=cloud[i].cpu().numpy(),
                              gt_part_r=gt_part_r[i].cpu().numpy(),
                              gt_part_quat=gt_part_quat[i].cpu().numpy(),
                              gt_part_t=gt_part_t[i].cpu().numpy(),
                              gt_joint_state=gt_joint_state[i].cpu().numpy(),
                              gt_norm_joint_loc=gt_norm_joint_loc[i].cpu().numpy(),
                              gt_norm_joint_axis=gt_norm_joint_axis[i].cpu().numpy(),
                              gt_norm_part_kp=gt_norm_part_kp[i].cpu().numpy(),
                              gt_scale=gt_scale[i].cpu().numpy(),
                              gt_center=gt_center[i].cpu().numpy(),
                              gt_norm_part_corners=gt_norm_part_corners[i].cpu().numpy(),
                              cate=cate[i].cpu().numpy(),
                              urdf_id=urdf_id[i].cpu().numpy())
                sample_id += 1
                results.append(result)

        print('\nwriting results to {}'.format(output_path))
        mmcv.dump(results, output_path)

    if opt.shuffle:
        np.random.shuffle(results)

    # multi-process eval
    num_process = opt.num_process
    ctx = mp.get_context('spawn')
    cpu_device = torch.device('cpu')
    params_dict = torch.load(osp.join(opt.params_dir, 'params.pth'))
    model = OMAD_Net(device=cpu_device, params_dict=params_dict,
                       num_points=opt.num_points, num_kp=opt.num_kp, num_parts=opt.num_parts,
                       init_dense_soft_factor=opt.dense_soft_factor, num_basis=opt.num_basis, symtype=opt.symtype)

    processes = []
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    gpu_ids = [id for id in range(num_gpus)]
    print('Using {} processes!'.format(num_process))
    chunk_size = int(len(results) / num_process) + 1
    split_results = [results[i:i + chunk_size] for i in range(0, len(results), chunk_size)]
    if not opt.show and opt.use_gpu:
        device_list = [gpu_ids[i % num_gpus] for i in range(num_process)]
        model_list = dict()
        for id in gpu_ids:
            model_list[id] = copy.deepcopy(model).to(torch.device('cuda:{}'.format(id)))
            model_list[id].set_device(torch.device('cuda:{}'.format(id)))
            model_list[id].share_memory()
    pid = os.getpid()
    for rank in range(num_process):
        file_name = osp.join(opt.work_dir, 'tmp_results', '{}process_final_result_{}_{}.pkl'.format(
            opt.num_process, opt.data_postfix, rank))
        if opt.use_cache and osp.exists(file_name):
            continue
        print('Create process rank {}!'.format(rank))

        if not opt.show and opt.use_gpu:
            new_model = model_list[device_list[rank]]
            new_device = torch.device('cuda:{}'.format(device_list[rank]))
            assert new_model.device == new_device
        else:
            new_model = model.share_memory()
            new_device = cpu_device
        p = ctx.Process(target=parallel_eval,
                        args=(pid, rank, split_results[rank], new_model, opt, new_device))
        p.daemon = True
        processes.append(p)
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    final_eval_results = collect_results(opt)

    r_errs_all = []
    t_errs_all = []
    ious_all = []
    joint_loc_errs_all = []
    joint_axis_errs_all = []
    joint_state_errs_all = []
    for sample_id, result_dict in final_eval_results.items():
        r_errs_all.append(np.array(result_dict['r_err']))
        t_errs_all.append(np.array(result_dict['t_err']))
        ious_all.append(np.array(result_dict['iou_3d']))
        joint_loc_errs_all.append(np.array(result_dict['joint_loc_err']))
        joint_axis_errs_all.append(np.array(result_dict['joint_axis_err']))
        joint_state_errs_all.append(np.array(result_dict['joint_state_err']))
    r_errs_all = np.stack(r_errs_all, axis=0)
    t_errs_all = np.stack(t_errs_all, axis=0)
    ious_all = np.stack(ious_all, axis=0)
    joint_loc_errs_all = np.stack(joint_loc_errs_all, axis=0)
    joint_axis_errs_all = np.stack(joint_axis_errs_all, axis=0)
    joint_state_errs_all = np.stack(joint_state_errs_all, axis=0)

    r_errs = [_ for _ in range(opt.num_parts)]
    for part_idx in range(opt.num_parts):
        idxs = ~ np.isnan(r_errs_all[:, part_idx])
        r_errs[part_idx] = np.mean(r_errs_all[idxs, part_idx])

    t_errs = [np.mean(t_errs_all[:, part_idx], axis=0) for part_idx in range(opt.num_parts)]
    ious = [np.mean(ious_all[:, part_idx], axis=0) for part_idx in range(opt.num_parts)]
    num_joints = opt.num_parts - 1
    joint_loc_errs = [np.mean(joint_loc_errs_all[:, joint_idx], axis=0) for joint_idx in range(num_joints)]

    joint_axis_errs = [_ for _ in range(num_joints)]
    for joint_idx in range(num_joints):
        idxs = ~ np.isnan(joint_axis_errs_all[:, joint_idx])
        joint_axis_errs[joint_idx] = np.mean(joint_axis_errs_all[idxs, joint_idx])
    joint_state_errs = [np.mean(joint_state_errs_all[:, joint_idx], axis=0) for joint_idx in range(num_joints)]
    print('mean r_errors:{}'.format(r_errs))
    print('mean t_errors:{}'.format(t_errs))
    print('mean iou_3d:{}'.format(ious))
    print('mean joint_loc_errors:{}'.format(joint_loc_errs))
    print('mean joint_axis_errors:{}'.format(joint_axis_errs))
    print('mean joint_state_errors:{}'.format(joint_state_errs))

