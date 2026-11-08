import numpy 
import heapq
import scene 
import torch 

class TopKFar:
    def __init__(self, k):
        self.k = k
        self.heap = []  # this will hold the k largest elements

    def push(self, cam_name, dist):
        if len(self.heap) < self.k:
            # Heap not full: simply push the item.
            heapq.heappush(self.heap, (cam_name, dist))
        else:
            # Heap is full: if the new item is larger than the smallest in the heap, replace it.
            if dist > self.heap[0][1]:
                heapq.heapreplace(self.heap, (cam_name, dist))

    def get_top_k(self):
        # Returns the k largest elements in descending order.
        return sorted(self.heap, key=lambda x: x[1], reverse=True)
    
def get_farest_cam_dict(scene):
    all_train_cameras = scene.getTrainCameras()
        
    cam_dict = {}
    all_cam_dict = {}
    for cam in all_train_cameras:
        all_cam_dict[cam.image_name] = cam
    
    reward_cam_dict = {}   
    for cam in all_train_cameras:
        cam_dict[cam.image_name] = []
        cam_center = cam.camera_center 
                
        tmp_top_queue = TopKFar(2)
        for other_cam in all_train_cameras:
            dist = torch.sum(torch.abs(other_cam.camera_center - cam_center))
            
            if other_cam.image_name != cam.image_name:
                tmp_top_queue.push(other_cam.image_name, dist)
        reward_cam_dict[cam.image_name] = [all_cam_dict[pair[0]] for pair in tmp_top_queue.get_top_k()] 
    return reward_cam_dict






    # for rl_densi_cnt in [2]: # 2 50 10
    #     for densi_init_std in [-1.0]:
    #         for scene in scene_list:
    #             for reward_scale_final in [4.0, 12.0]: # 4.0 to small ?
    #                 for run_id in [1,2]:
    #                     for pl_lr_final in [0.00001]: # 0.0001 -> 0.00001
    #                         for reward_scale_v2 in [10.0]: # 10.0
    #                             for nv_iters in [10000]:
    #                                 for grdclip in [1.2]: # default 1.2 change to 
    #                                     for init_std in [-1.0]: # 0.0 good ,but  -1 is btter ?
                                    
changed_list = ["rl_densi_cnt", "init_std", "densi_init_std", "reward_scale_final", "run_id", "pl_lr_final", "nv_iters"]

def generate_model_name(opt, defaults, base_name="base_model"):
    """
    Generate a model name suffix based on non-default option values.
    
    Parameters:
        opt (Namespace): The optimization parameters as an object (e.g. a namespace).
        defaults (dict): A dictionary of default values for each option.
        base_name (str): The starting base name for the model.
    
    Returns:
        str: The model name that includes a suffix for each non-default parameter.
    """
    # Start with the base model name.
    model_name = base_name
    changed_dict = {}
    
    # Use vars(opt) to get the dictionary of parameters and their values.
    for key, value in vars(opt).items():
        # Compare the current value with the default.
        default_value = defaults.get(key)
        if value != default_value:
            # Append the parameter that was changed to the model name.
            model_name += f"-{key[0:3]}_{value}"
            changed_dict[key] = value 
    return model_name, changed_dict