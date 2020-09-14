import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from dlt import *


def compute_mpjpe(output, target):
    return torch.mean(torch.sqrt(torch.sum((output-target)*(output-target), -1)))


def get_projection(az, el, distance, focal_length=35, img_w=256, img_h=256, sensor_size_mm = 32.):
    """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""

    # Calculate intrinsic matrix.
    f_u = focal_length * img_w  / sensor_size_mm
    f_v = focal_length * img_h  / sensor_size_mm
    u_0 = img_w / 2
    v_0 = img_h / 2
    K = np.matrix(((f_u, 0, u_0), (0, f_v, v_0), (0, 0, 1)))

    # Calculate rotation and translation matrices.
    sa = np.sin(np.radians(-az))
    ca = np.cos(np.radians(-az))
    se = np.sin(np.radians(-el))
    ce = np.cos(np.radians(-el))
    R_world2cam = np.transpose(np.matrix(((ca * ce, -sa, ca * se),
                                          (sa * ce, ca, sa * se),
                                          (-se, 0, ce))))

    T_world2cam = np.transpose(np.matrix((0,
                                           0,
                                           distance)))
    RT = np.hstack((R_world2cam, T_world2cam))

    return K*RT


def generate_data(batch_size, number_of_cameras):
    "Generate data for benchamrk."
    # generate batch_size random 3D points around origin
    points_3d = torch.rand(batch_size, 3)-0.5
    # generate number_of_cameras camera view points
    projections = torch.stack([torch.tensor(get_projection( 360*np.random.rand(), 45*np.random.rand(), 0.5*np.random.rand() + 2.0)).float() for i in range(number_of_cameras)])
    projections = projections.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    # project points to image space
    points_2d = project(points_3d, projections)
    return points_3d, projections, points_2d

if __name__ == "__main__":

    # first generate syntehtic data
    number_of_joints = 17
    batch_size = 256*number_of_joints
    number_of_cameras = 4
    points_3d, projections, points_2d = generate_data(batch_size, number_of_cameras)

    # produce Figure 3b of the manuscript
    sigmas = np.linspace(0, 10, num=10)
    mpjpe_2d = []
    mpjpe_3d_svd = []
    mpjpe_3d_1iter = []
    mpjpe_3d_2iter = []
    mpjpe_3d_3iter = []

    print("Computing accuracies...")

    for sigma in sigmas:

        points_2d_noisy = points_2d + sigma*torch.randn(points_2d.shape)
        points_3d_dlt_svd = triangulate_from_multiple_views_svd(projections, points_2d_noisy)
        points_3d_dlt_1iter = triangulate_from_multiple_views_sii(projections, points_2d_noisy, number_of_iterations = 1)
        points_3d_dlt_2iter = triangulate_from_multiple_views_sii(projections, points_2d_noisy, number_of_iterations = 2)
        points_3d_dlt_3iter = triangulate_from_multiple_views_sii(projections, points_2d_noisy, number_of_iterations = 3)

        mpjpe_2d.append(compute_mpjpe(points_2d_noisy, points_2d).detach().cpu().numpy().item())
        mpjpe_3d_svd.append(compute_mpjpe(points_3d_dlt_svd, points_3d).detach().cpu().numpy().item())
        mpjpe_3d_1iter.append(compute_mpjpe(points_3d_dlt_1iter, points_3d).detach().cpu().numpy().item())
        mpjpe_3d_2iter.append(compute_mpjpe(points_3d_dlt_2iter, points_3d).detach().cpu().numpy().item())
        mpjpe_3d_3iter.append(compute_mpjpe(points_3d_dlt_3iter, points_3d).detach().cpu().numpy().item())

    print("3D-MPJPE SVD: ", mpjpe_3d_svd)
    print("3D-MPJPE SII-1: ", mpjpe_3d_1iter)
    print("3D-MPJPE SII-2: ", mpjpe_3d_2iter)
    print("3D-MPJPE SII-3: ", mpjpe_3d_3iter)
    print("2D-MPJPE: ", mpjpe_2d)
    print("=================================")

    plt.figure()
    plt.plot(mpjpe_2d, mpjpe_3d_svd, label='SVD')
    plt.plot(mpjpe_2d, mpjpe_3d_1iter, label='SII-1')
    plt.plot(mpjpe_2d, mpjpe_3d_2iter, label='SII-2')
    plt.plot(mpjpe_2d, mpjpe_3d_3iter, label='SII-3')
    plt.grid()
    plt.legend()
    plt.xlabel("MPJPE-2D")
    plt.ylabel("MPJPE-3D")
    plt.title("Accuracy")
    plt.savefig("./output/accuracy.png")


    print("Timing on GPU...")

    # produce Figure 3d of the manuscript
    number_of_measurements = 10
    batch_sizes = [16, 32, 64, 128, 256, 512]
    time_svd = []
    time_1iter = []
    time_2iter = []
    time_3iter = []
    # fixed amount of noise, doesn't matter here anyway
    points_2d_noisy = points_2d + 2*torch.randn(points_2d.shape)

    for batch_size in batch_sizes:
        t_svd = 0
        t_1iter = 0
        t_2iter = 0
        t_3iter = 0
        for i in range(number_of_measurements):
            start = time.time()
            points_3d_dlt_svd = triangulate_from_multiple_views_svd(projections[0:batch_size*number_of_joints].cuda(), points_2d_noisy[0:batch_size*number_of_joints].cuda())
            end = time.time()
            t_svd+= end-start

            start = time.time()
            points_3d_dlt_1iter = triangulate_from_multiple_views_sii(projections[0:batch_size*number_of_joints].cuda(), points_2d_noisy[0:batch_size*number_of_joints].cuda(), number_of_iterations = 1)
            end = time.time()
            t_1iter+= end-start

            start = time.time()
            points_3d_dlt_2iter = triangulate_from_multiple_views_sii(projections[0:batch_size*number_of_joints].cuda(), points_2d_noisy[0:batch_size*number_of_joints].cuda(), number_of_iterations = 2)
            end = time.time()
            t_2iter+= end-start

            start = time.time()
            points_3d_dlt_3iter = triangulate_from_multiple_views_sii(projections[0:batch_size*number_of_joints].cuda(), points_2d_noisy[0:batch_size*number_of_joints].cuda(), number_of_iterations = 3)
            end = time.time()
            t_3iter+= end-start

        time_svd.append(t_svd/number_of_measurements)
        time_1iter.append(t_1iter/number_of_measurements)
        time_2iter.append(t_2iter/number_of_measurements)
        time_3iter.append(t_3iter/number_of_measurements)

    print("GPU time SVD: ", time_svd)
    print("GPU time SII-1: ", time_1iter)
    print("GPU time SII-2: ", time_2iter)
    print("GPU time SII-3: ", time_3iter)
    print("Batch sizes: ", batch_sizes)
    print("=================================")

    plt.figure()
    plt.semilogy(batch_sizes, time_svd, label='SVD')
    plt.semilogy(batch_sizes, time_1iter, label='SII-1')
    plt.semilogy(batch_sizes, time_2iter, label='SII-2')
    plt.semilogy(batch_sizes, time_3iter, label='SII-3')
    plt.grid(True)
    plt.legend()
    plt.xlabel("Batch size")
    plt.ylabel("Time")
    plt.title("GPU Runtime")
    plt.savefig("./output/time_gpu.png")
