import torch

def homogeneous_to_euclidean(points):
    """Converts torch homogeneous points to euclidean
    Args:
        points torch tensor of shape (N, M + 1): N homogeneous points of dimension M
    Returns:
        torch tensor of shape (N, M): euclidean points
    """
    if torch.is_tensor(points):
        return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(1, 0)
    else:
        raise TypeError("This methods expects a PyTorch tensor.")

def euclidean_to_homogeneous(points):
    """Converts torch euclidean points to homogeneous
    Args:
        points torch tensor of shape (N, M): N euclidean points of dimension M
    Returns:
        torch tensor of shape (N, M+1): homogeneous points
    """
    if torch.is_tensor(points):
        return torch.cat((points, torch.ones(points.shape[0], 1).to(points.device)),-1)
    else:
        raise TypeError("This methods expects a PyTorch tensor.")

def project(points, projections):
    """Project batch of 3D points to 2D
    Args:
        points torch tensor of shape (B, 3)
        projections torch tensor of shape (B, N, 3, 4)
    Returns:
        torch tensor of shape (B, N, 2)
    """
    points_homogeneous = euclidean_to_homogeneous(points)
    points_homogeneous = points_homogeneous.unsqueeze(1).repeat(1, projections.shape[1], 1)
    points_2d_homogeneous = torch.matmul(projections.reshape(-1,3,4), points_homogeneous.reshape(-1,4,1)).unsqueeze(-1)
    points_2d = homogeneous_to_euclidean(points_2d_homogeneous)
    return points_2d.reshape(projections.shape[0], projections.shape[1], 2)


def triangulate_from_multiple_views_sii(proj_matricies, points, number_of_iterations = 2):
    """This module lifts B 2d detections obtained from N viewpoints to 3D using the Direct Linear Transform method.
    It computes the eigenvector associated to the smallest eigenvalue using the Shifted Inverse Iterations algorithm.
    Args:
        proj_matricies torch tensor of shape (B, N, 3, 4): sequence of projection matricies (3x4)
        points torch tensor of of shape (B, N, 2): sequence of points' coordinates
    Returns:
        point_3d torch tensor of shape (B, 3): triangulated points
    """

    batch_size = proj_matricies.shape[0]
    n_views = proj_matricies.shape[1]

    # assemble linear system
    A = proj_matricies[:,:, 2:3].expand(batch_size, n_views, 2, 4) * points.view(-1, n_views, 2, 1)
    A -= proj_matricies[:, :, :2]
    A = A.view(batch_size, -1, 4)

    AtA = A.permute(0,2,1).matmul(A).float()
    I = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1).to(A.device)
    B =  AtA + 0.001*I
    # initialize normalized random vector
    bk = torch.rand(batch_size, 4, 1).float().to(AtA.device)
    norm_bk = torch.sqrt(bk.permute(0,2,1).matmul(bk))
    bk = bk/norm_bk
    for k in range(number_of_iterations):
        bk, _ = torch.solve(bk, B)
        norm_bk = torch.sqrt(bk.permute(0,2,1).matmul(bk))
        bk = bk/norm_bk

    point_3d_homo = -bk.squeeze(-1)
    point_3d = homogeneous_to_euclidean(point_3d_homo)

    return point_3d

def triangulate_from_multiple_views_svd(proj_matricies, points):
    """This module lifts B 2d detections obtained from N viewpoints to 3D using the Direct Linear Transform method.
    It computes the eigenvector associated to the smallest eigenvalue using Singular Value Decomposition.
    Args:
        proj_matricies torch tensor of shape (B, N, 3, 4): sequence of projection matricies (3x4)
        points torch tensor of of shape (B, N, 2): sequence of points' coordinates
    Returns:
        point_3d numpy torch tensor of shape (B, 3): triangulated point
    """

    batch_size = proj_matricies.shape[0]
    n_views = proj_matricies.shape[1]

    A = proj_matricies[:,:, 2:3].expand(batch_size, n_views, 2, 4) * points.view(-1, n_views, 2, 1)
    A -= proj_matricies[:, :, :2]

    #_, _, vh = torch.svd(A.view(batch_size, -1, 4))
    _, _, vh = torch.svd(A.view(batch_size, -1, 4))

    point_3d_homo = -vh[:, :, 3]
    point_3d = homogeneous_to_euclidean(point_3d_homo)

    return point_3d
