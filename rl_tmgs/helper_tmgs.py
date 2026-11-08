import random 
from utils.taming_utils import compute_gaussian_score
def densify_iteration(
    scene,
    all_edges,
    cams,
    gaussians,
    pipe,
    bg,
    score_coefficients,
    opt,
    size_threshold,
    counts_array,
    radii,
    iter_num,
):
    """
    Perform one densification step over a random subset of train cameras.

    Parameters
    ----------
    scene : Scene
        Your scene object, must implement getTrainCameras() and have cameras_extent.
    all_edges : Sequence
        List/array of edge losses corresponding to each camera.
    cams : int
        Number of cameras to sample; if -1, sample all.
    gaussians : GaussianField
        The gaussian field to densify (mutated in place).
    pipe, bg, score_coefficients, opt
        Additional objects needed by compute_gaussian_score.
    size_threshold : float
        max_screen_size passed to densify_with_score.
    counts_array : Sequence[int]
        Budgets for each iteration.
    radii : Sequence[float]
        Radii for each gaussian.
    iter_num : int
        Current iteration index (used for choosing budget and then incremented).

    Returns
    -------
    int
        iter_num + 1
    """
    # copy the stacks so we don’t destroy the originals
    viewpoints = scene.getTrainCameras().copy()
    edges = all_edges.copy()

    # decide how many cams to draw
    num_cams = len(viewpoints) if cams == -1 else cams

    # sample without replacement
    cam_list = []
    edge_losses = []
    for _ in range(num_cams):
        idx = random.randrange(len(viewpoints))
        cam_list.append(viewpoints.pop(idx))
        edge_losses.append(edges.pop(idx))

    # compute importance scores and densify
    gaussian_importance = compute_gaussian_score(
        scene, cam_list, edge_losses,
        gaussians, pipe, bg,
        score_coefficients, opt
    )
    gaussians.densify_with_score(
        scores=gaussian_importance,
        max_screen_size=size_threshold,
        min_opacity=0.005,
        extent=scene.cameras_extent * opt.densi_scaler * opt.cameras_extent_scale,
        budget=counts_array[iter_num+1],
        radii=radii,
        iter_num=iter_num,
        grad_threshold=opt.densify_grad_threshold,
    )

    # advance the iteration counter
    return iter_num + 1





def densify_iteration_scaleup(
    scene,
    all_edges,
    cams,
    gaussians,
    pipe,
    bg,
    score_coefficients,
    opt,
    size_threshold,
    counts_array,
    radii,
    iter_num,
    resultkeys,
    densi_parameter_ori_dict,
):
    """
    Perform one densification step over a random subset of train cameras.

    Parameters
    ----------
    scene : Scene
        Your scene object, must implement getTrainCameras() and have cameras_extent.
    all_edges : Sequence
        List/array of edge losses corresponding to each camera.
    cams : int
        Number of cameras to sample; if -1, sample all.
    gaussians : GaussianField
        The gaussian field to densify (mutated in place).
    pipe, bg, score_coefficients, opt
        Additional objects needed by compute_gaussian_score.
    size_threshold : float
        max_screen_size passed to densify_with_score.
    counts_array : Sequence[int]
        Budgets for each iteration.
    radii : Sequence[float]
        Radii for each gaussian.
    iter_num : int
        Current iteration index (used for choosing budget and then incremented).

    Returns
    -------
    int
        iter_num + 1
    """
    # copy the stacks so we don’t destroy the originals
    viewpoints = scene.getTrainCameras().copy()
    edges = all_edges.copy()

    # decide how many cams to draw
    num_cams = len(viewpoints) if cams == -1 else cams

    # sample without replacement
    cam_list = []
    edge_losses = []
    for _ in range(num_cams):
        idx = random.randrange(len(viewpoints))
        cam_list.append(viewpoints.pop(idx))
        edge_losses.append(edges.pop(idx))

    # compute importance scores and densify
    gaussian_importance = compute_gaussian_score(
        scene, cam_list, edge_losses,
        gaussians, pipe, bg,
        score_coefficients, opt
    )
    # densi_adjust_dict[k]['action'].item()
    
    if resultkeys is not None:
        for k in score_coefficients:
            if k in resultkeys:
                score_coefficients[k] = float(resultkeys[k]['action'].item()) * densi_parameter_ori_dict[k]
    gaussians.densify_with_score(
        scores=gaussian_importance,
        max_screen_size=size_threshold,
        min_opacity=0.005,
        extent=scene.cameras_extent * opt.densi_scaler,
        budget=counts_array[iter_num+1],
        radii=radii,
        iter_num=iter_num
    )

    # advance the iteration counter
    return iter_num + 1
