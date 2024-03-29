"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.
"""
# Modified from smplx code for FLAME
import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F
from pprint import pprint


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


def vertices2joints(J_regressor, vertices):
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def blend_shapes(betas, shape_disps):
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(rot_mats.view(-1, joints.shape[1], 3, 3),
                                   rel_joints.view(-1, joints.shape[1], 3,
                                                   1)).view(
                                                       -1, joints.shape[1], 4,
                                                       4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:,
                                                                            i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


def lbs(shape_params,
        expression_params,
        pose,
        v_template,
        shapedirs,
        posedirs,
        J_regressor,
        parents,
        lbs_weights,
        pose2rot=True,
        dtype=torch.float32):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''
    betas = torch.cat([shape_params, expression_params], dim=1)

    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3),
                                   dtype=dtype).view([batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, posedirs) \
            .view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype,
                               device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed


class FLAME(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(self, config):
        super(FLAME, self).__init__()
        print("creating the FLAME Decoder")
        self.config = config

        with open(self.config.flame_model_path, 'rb') as f:
            # flame_model = Struct(**pickle.load(f, encoding='latin1'))
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)
        self.param2verts = lbs
        tp = np.load(self.config.trim_path, allow_pickle=True)

        self.dtype = torch.float32
        self.register_buffer(
            'faces_tensor',
            to_tensor(to_np(
                tp['map_verts'][flame_model.f[tp['idx_faces']]].copy(),
                dtype=np.int64),
                      dtype=torch.long))
        # The vertices of the template model
        self.register_buffer(
            'v_template',
            to_tensor(to_np(flame_model.v_template[tp['idx_verts']].copy()),
                      dtype=self.dtype))
        # The shape components and expression
        shapedirs = to_tensor(to_np(
            flame_model.shapedirs[tp['idx_verts']].copy()),
                              dtype=self.dtype)
        shapedirs = torch.cat([
            shapedirs[:, :, :self.config.shape_params],
            shapedirs[:, :, 300:300 + self.config.expression_params]
        ], 2)

        self.register_buffer('shapedirs', shapedirs)

        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs[tp['idx_verts']].copy(),
                              [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=self.dtype))
        #
        self.register_buffer(
            'J_regressor',
            to_tensor(to_np(flame_model.J_regressor[:, tp['idx_verts']].copy()),
                      dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer(
            'lbs_weights',
            to_tensor(to_np(flame_model.weights[tp['idx_verts']].copy()),
                      dtype=self.dtype))

        # Fixing Eyeball and neck rotation
        default_eyball_pose = torch.zeros([1, 6],
                                          dtype=self.dtype,
                                          requires_grad=False)
        self.register_parameter(
            'eye_pose', nn.Parameter(default_eyball_pose, requires_grad=False))
        default_neck_pose = torch.zeros([1, 3],
                                        dtype=self.dtype,
                                        requires_grad=False)
        self.register_parameter(
            'neck_pose', nn.Parameter(default_neck_pose, requires_grad=False))

        # Static and Dynamic Landmark embeddings for FLAME
        lmk_embeddings = np.load(self.config.flame_lmk_embedding_path,
                                 allow_pickle=True,
                                 encoding='latin1')
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer(
            'lmk_faces_idx',
            torch.tensor(
                tp['map_faces'][lmk_embeddings['static_lmk_faces_idx']].copy(),
                dtype=torch.long))
        self.register_buffer(
            'lmk_bary_coords',
            torch.tensor(lmk_embeddings['static_lmk_bary_coords'],
                         dtype=self.dtype))
        self.register_buffer(
            'dynamic_lmk_faces_idx',
            torch.tensor(
                tp['map_faces'][lmk_embeddings['dynamic_lmk_faces_idx']].copy(),
                dtype=torch.long))
        self.register_buffer(
            'dynamic_lmk_bary_coords',
            torch.tensor(lmk_embeddings['dynamic_lmk_bary_coords'],
                         dtype=self.dtype))
        self.register_buffer(
            'full_lmk_faces_idx',
            torch.tensor(
                tp['map_faces'][lmk_embeddings['full_lmk_faces_idx']].copy(),
                dtype=torch.long))
        self.register_buffer(
            'full_lmk_bary_coords',
            torch.tensor(lmk_embeddings['full_lmk_bary_coords'],
                         dtype=self.dtype))

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))

    def vertices2landmarks(self, vertices, faces, lmk_faces_idx,
                           lmk_bary_coords):
        ''' Calculates landmarks by barycentric interpolation

            Parameters
            ----------
            vertices: torch.tensor BxVx3, dtype = torch.float32
                The tensor of input vertices
            faces: torch.tensor Fx3, dtype = torch.long
                The faces of the mesh
            lmk_faces_idx: torch.tensor L, dtype = torch.long
                The tensor with the indices of the faces used to calculate the
                landmarks.
            lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
                The tensor of barycentric coordinates that are used to interpolate
                the landmarks

            Returns
            -------
            landmarks: torch.tensor BxLx3, dtype = torch.float32
                The coordinates of the landmarks for each mesh in the batch
        '''
        # Extract the indices of the vertices for each face
        # BxLx3
        batch_size, num_verts = vertices.shape[:2]
        device = vertices.device

        lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
            batch_size, -1, 3)

        lmk_faces += torch.arange(batch_size, dtype=torch.long,
                                  device=device).view(-1, 1, 1) * num_verts

        lmk_vertices = vertices.view(-1,
                                     3)[lmk_faces].view(batch_size, -1, 3, 3)

        landmarks = torch.einsum('blfi,blf->bli',
                                 [lmk_vertices, lmk_bary_coords])
        return landmarks

    def _find_dynamic_lmk_idx_and_bcoords(self,
                                          cam,
                                          dynamic_lmk_faces_idx,
                                          dynamic_lmk_b_coords,
                                          neck_kin_chain,
                                          dtype=torch.float32):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
        """

        batch_size = cam.shape[0]
        angles = cam[:, :3]
        y_rot_angle = torch.round(
            torch.clamp(torch.abs(angles[:, 1]) * 180.0 / np.pi,
                        max=39)).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx, 0,
                                               y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0,
                                              y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def seletec_3d68(self, vertices):
        landmarks3d = self.vertices2landmarks(
            vertices, self.faces_tensor,
            self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
            self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1))
        return landmarks3d

    def forward(self,
                shape_params=None,
                expression_params=None,
                pose_params=None,
                eye_pose_params=None,
                cam_params=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters (6)
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1)
        pose0 = torch.zeros_like(pose_params[:, :3])
        full_pose = torch.cat([
            pose0,
            self.neck_pose.expand(batch_size, -1), pose_params, eye_pose_params
        ],
                              dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(
            batch_size, -1, -1)

        vertices, _ = self.param2verts(shape_params,
                                       expression_params,
                                       full_pose,
                                       template_vertices,
                                       self.shapedirs,
                                       self.posedirs,
                                       self.J_regressor,
                                       self.parents,
                                       self.lbs_weights,
                                       dtype=self.dtype)

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(
            batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(
            batch_size, -1, -1)

        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            cam_params,
            self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain,
            dtype=self.dtype)
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks2d = self.vertices2landmarks(vertices, self.faces_tensor,
                                              lmk_faces_idx, lmk_bary_coords)
        bz = vertices.shape[0]
        landmarks3d = self.vertices2landmarks(
            vertices, self.faces_tensor, self.full_lmk_faces_idx.repeat(bz, 1),
            self.full_lmk_bary_coords.repeat(bz, 1, 1))

        return vertices, landmarks2d, landmarks3d


class FLAMETex(nn.Module):
    """
    current FLAME texture are adapted from BFM Texture Model
    """

    def __init__(self, config):
        super(FLAMETex, self).__init__()
        self.config = config
        tex_params = self.config.tex_params
        tex_space = np.load(self.config.tex_space_path, allow_pickle=True)
        texture_mean = tex_space['mean'].reshape(1, -1)
        texture_basis = tex_space['tex_dir'].reshape(-1, 200)
        num_components = texture_basis.shape[1]
        texture_mean = torch.from_numpy(texture_mean).float()[None, ...]
        texture_basis = torch.from_numpy(
            texture_basis[:, :tex_params]).float()[None, ...]
        self.register_buffer('texture_mean', texture_mean)
        self.register_buffer('texture_basis', texture_basis)

    def forward(self, texcode):
        texture = self.texture_mean + (self.texture_basis *
                                       texcode[:, None, :]).sum(-1)
        texture = texture.reshape(texcode.shape[0], 512, 512,
                                  3).permute(0, 3, 1, 2)
        texture = F.interpolate(texture, [256, 256])
        texture = texture[:, [2, 1, 0], :, :]
        return texture
