import os 

DATASETROOT =  "/data/zhan_li/processed_datasets"

DATA = {}
PYTHON_PATH = "/opt/conda/envs/mcmc/bin/python"
SAVEROOT = "/data/zhan_li/weight/gsrl/aaai"

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
mipnerfmcmc =  ["bicycle", "garden", "stump","room", "counter", "kitchen", "bonsai"]
miproot = f"{DATASETROOT}/perscene/mipnerf"


tandtroot =  f"{DATASETROOT}/perscene/tandt/"
tandts = ["train", "truck"]
db = ["drjohnson", "playroom"]
dbroot =f"{DATASETROOT}/perscene/db"

benchmarkdb = ["drjohnson", "playroom"]
benchmark1 = ["train", "truck","drjohnson", "playroom"]
benchmarktt = ["train", "truck"]
for k in db:
    DATA[k] = os.path.join(dbroot, k)
for k in tandts:
    DATA[k] = os.path.join(tandtroot, k)
for k in mipnerf360_outdoor_scenes:
    DATA[k] = os.path.join(miproot, k)
    
for k in mipnerf360_indoor_scenes:
    DATA[k] = os.path.join(miproot, k) 


# 1_610_000
# tmgs_budget = {
#     "bicycle":   5_987_095,
#     "flowers":   3_618_411,
#     "garden":    5_728_191,
#     "stump":     4_867_429,
#     "treehill":  3_770_257,
#     "room":      1_548_960,
#     "counter":   1_190_919,
#     "kitchen":   1_803_735,
#     "playroom":  2_326_100,
#     "bonsai":    1_252_367,
#     "truck":     2_584_171,
#     "train":     1_085_480,
#     "drjohnson": 3_273_600,
# }


tmgs_budget = {
    "bicycle":   5_987_095,
    "flowers":   3_618_411,
    "garden":    5_070_000,
    "stump":     4_867_429,
    "treehill":  3_770_257,
    "room":      1_548_960,
    "counter":   1_190_919,
    "kitchen":   1_610_000,
    "playroom":  2_326_100,
    "bonsai":    1_190_000,
    "truck":     2_584_171,
    "train":     1_085_480,
    "drjohnson": 3_273_600,
}
tmgs_budget_scaler = {}

# for idx in range(0,5):
#     tmgs_budget_scaler[idx]= {}

# tmgs_budget_scaler[0]= []
# tmgs_budget_scaler[1]= []
# tmgs_budget_scaler[2]= []
# tmgs_budget_scaler[3]= []
# tmgs_budget_scaler[4]= []

tmgs_budget2 = {
    "bicycle":   5_987_095,
    "flowers":   3_618_411,
    "garden":    5_728_191,
    "stump":     4_867_429,
    "treehill":  3_770_257,
    "room":      1_548_960,
    "counter":   1_190_919,
    "kitchen":   1_803_735,
    "playroom":  1_400_000,
    "bonsai":    1_252_367,
    "truck":     2_584_171,
    "train":     1_085_480,
    "drjohnson": 3_273_600,
}



tmgs_budget_small = {
    "bicycle": 1.2,
    "flowers": 0.72,
    "garden": 1.15,
    "stump": 0.97,
    "treehill": 0.75,
    "room": 0.31,
    "counter": 0.24,
    "kitchen": 0.36,
    "bonsai": 0.25,
    "truck": 0.52,
    "train": 0.22,
    "playroom": 0.47,
    "drjohnson": 0.65
}

dl3dv_example =['5c3af5', '5f0041', '800cf8', '9c8c0e', 'ba55c8', 'cbd44b', 'cc08c0', 'cd9c98', 'dac979', 'dafa9c', 'e8ce51']

dl3dv_root = f"{DATASETROOT}/perscene/dl3dv/DL3DV-10K-Benchmark"
dl3dv_buggy = ["cbd44b","ba55c8"]

for k in dl3dv_example:
    DATA[k]= os.path.join(dl3dv_root, k)
    