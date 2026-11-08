#!/usr/bin/env python3
"""Portable setup/check/evaluation entry points; never launch training implicitly."""
import argparse
import ast
import datetime
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("rl_tmgs", "rl_asymgs", "rl_mcmc")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(command, cwd, env=None, dry=False):
    import shlex
    print("+ " + shlex.join([str(x) for x in command]), flush=True)
    if not dry:
        subprocess.run([str(x) for x in command], cwd=cwd, env=env, check=True)


def conda_executable():
    value = os.environ.get("CONDA_EXE") or shutil.which("conda")
    require(value, "Conda was not found. Install Miniconda/Miniforge or set CONDA_EXE.")
    return value


def prefix_for(args, config):
    if args.prefix:
        return Path(args.prefix).expanduser().resolve()
    name = args.env or config["environment"]
    info = json.loads(subprocess.check_output([conda_executable(), "info", "--json"], text=True))
    existing = [Path(x) for x in info["envs"] if Path(x).name == name]
    require(len(existing) < 2, "Environment name is ambiguous; pass --prefix.")
    if existing:
        return existing[0]
    return Path(info["envs_dirs"][0]) / name


def python_for(args, config):
    if args.python:
        value = shutil.which(args.python) or args.python
        path = Path(value).expanduser().absolute()
    else:
        path = prefix_for(args, config) / "bin" / "python"
    require(path.is_file(), f"Python not found: {path}. Run setup.sh or pass --python /path/to/python.")
    return path


def environment(method):
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", MKL_THREADING_LAYER="GNU", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    env["PYTHONPATH"] = str(method) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if method.name == "rl_asymgs":
        env["NERFBASELINES_REGISTER"] = str(method / "asym_emags_spec.py")
        env["NERFBASELINES_BACKEND"] = "python"
    return env


def setup(args, method, config):
    for extension in config["extensions"]:
        require((method / extension / "setup.py").is_file(), f"Missing extension source: {method / extension}")
    if args.check_only:
        check(args, method, config)
        return
    require(not args.python, "--python selects an existing interpreter for --check-only; use --env or --prefix for installation.")
    prefix = prefix_for(args, config)
    require(not prefix.exists(), f"Environment already exists: {prefix}. Use --check-only, or choose a new --env/--prefix. Existing environments are never reinstalled.")
    conda = conda_executable()
    run([conda, "env", "create", "--yes", "--prefix", prefix, "--file", method / "environment_rlgs.yml"], method, dry=args.dry_run)
    python = prefix / "bin/python"
    env = environment(method)
    env["CUDA_HOME"] = str(prefix)
    env["PATH"] = str(prefix / "bin") + os.pathsep + env.get("PATH", "")
    env["MAX_JOBS"] = str(args.jobs)
    compiler_prefix = prefix / "bin/x86_64-conda-linux-gnu"
    cc = Path(str(compiler_prefix) + "-gcc")
    cxx = Path(str(compiler_prefix) + "-g++")
    if not args.dry_run:
        require(cc.is_file() and cxx.is_file(), "Managed GCC/G++ compilers are missing from the new environment.")
    env["CC"] = str(cc)
    env["CXX"] = str(cxx)
    if not args.dry_run:
        version = subprocess.check_output([prefix / "bin/nvcc", "--version"], text=True)
        require("release 12.6," in version, "Expected CUDA Toolkit 12.6 in the new environment.")
    for extension in config["extensions"]:
        run([python, "-m", "pip", "install", "--no-build-isolation", "--no-deps", method / extension], method, env, args.dry_run)
    run([python, "-m", "pip", "check"], method, env, args.dry_run)
    command = [python, "-B", Path(__file__), method.name, "_probe"]
    if args.imports_only:
        command += ["--imports-only"]
    run(command, method, env, args.dry_run)


def check(args, method, config):
    python = python_for(args, config)
    command = [python, "-B", Path(__file__), method.name, "_probe"]
    if args.imports_only:
        command += ["--imports-only"]
    run(command, method, environment(method), args.dry_run)


def probe(args, method, config):
    sys.path.insert(0, str(method))
    expected = config["versions"]
    actual = {name: importlib.metadata.version(name) for name in expected}
    for name, value in expected.items():
        require(actual[name] == value, f"{name}: expected {value}, found {actual[name]}")
    require(sys.version_info[:2] == tuple(map(int, config["python"].split('.'))), f"Expected Python {config['python']}, found {sys.version.split()[0]}")
    import torch
    import numpy as np
    from simple_knn._C import distCUDA2
    from gaussian_renderer import GaussianModel, render
    from scene.cameras import MiniCam
    from utils.graphics_utils import BasicPointCloud, getProjectionMatrix
    from types import SimpleNamespace
    loaded = {}
    for name in config["imports"]:
        module = importlib.import_module(name)
        loaded[name] = getattr(module, "__file__", None)
    # Import the model/policy code too; no training or weight downloads happen here.
    for name in config["project_imports"]:
        importlib.import_module(name)
    if args.imports_only:
        print(json.dumps({"method": method.name, "status": "imports_only", "versions": actual, "modules": loaded}, indent=2))
        return
    require(torch.cuda.is_available(), "CUDA is unavailable; check the NVIDIA driver and selected Python environment.")
    torch.manual_seed(0)
    points = np.array([[x, y, z] for z in (2., 2.2) for y in (-.2, .2) for x in (-.2, .2)], dtype=np.float32)
    colors = np.tile(np.array([[.7, .3, .2]], dtype=np.float32), (len(points), 1))
    distances = distCUDA2(torch.tensor(points, device="cuda"))
    require(bool(torch.isfinite(distances).all() and (distances > 0).all()), "KNN kernel returned invalid distances.")
    if method.name == "rl_tmgs":
        model = GaussianModel(3, optimizer_type="default", rendering_mode="abs")
    else:
        model = GaussianModel(3)
    model.create_from_pcd(BasicPointCloud(points=points, colors=colors, normals=np.zeros_like(points)), 1.)
    if method.name == "rl_asymgs":
        model.filter_3D = torch.zeros((len(points), 1), device="cuda")
    view = torch.eye(4, device="cuda")
    projection = getProjectionMatrix(znear=.01, zfar=100., fovX=1., fovY=1.).transpose(0, 1).cuda()
    camera = MiniCam(32, 32, 1., 1., .01, 100., view, projection)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, separate_sh=True)
    kwargs = {"kernel_size": .1} if method.name == "rl_asymgs" else {}
    result = render(camera, model, pipe, torch.zeros(3, device="cuda"), **kwargs)
    output = result["render"]
    require(tuple(output.shape) == (3, 32, 32) and bool(torch.isfinite(output).all()), "Rasterizer returned an invalid image.")
    require(bool((result["radii"] > 0).any()) and float(output.sum()) > 0, "Smoke-test Gaussians were not rendered.")
    output.square().mean().backward()
    for name in ("_xyz", "_opacity", "_scaling"):
        gradient = getattr(model, name).grad
        require(gradient is not None and bool(torch.isfinite(gradient).all()), f"Invalid rasterizer gradient: {name}")
        require(bool((gradient != 0).any()), f"Rasterizer backward returned only zero gradients: {name}")
    if "fused_ssim" in config["imports"]:
        from fused_ssim import fused_ssim
        image = output.detach().unsqueeze(0).requires_grad_(True)
        value = fused_ssim(image, torch.zeros_like(image))
        value.backward()
        require(bool(torch.isfinite(value).all() and torch.isfinite(image.grad).all()), "Fused SSIM forward/backward failed.")
    torch.cuda.synchronize()
    print(json.dumps({"method": method.name, "status": "cuda_smoke_passed", "gpu": torch.cuda.get_device_name(), "versions": actual, "modules": loaded, "checks": ["project imports", "KNN", "rasterizer forward", "rasterizer backward"]}, indent=2))


def read_config(path):
    node = ast.parse(path.read_text(), mode="eval").body
    require(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Namespace" and not node.args, "cfg_args must contain a literal argparse Namespace.")
    require(all(x.arg for x in node.keywords), "Expanded cfg_args keywords are not supported.")
    return {item.arg: ast.literal_eval(item.value) for item in node.keywords}


def validate_metrics(path, method, iteration):
    require(path.is_file(), f"Evaluation did not produce {path}; upstream metrics scripts may return exit code 0 after an error.")
    data = json.loads(path.read_text())
    metrics = data.get("metrics", {}) if method.name == "rl_asymgs" else data.get(f"ours_{iteration}", {})
    required = ("psnr", "ssim", "lpips") if method.name == "rl_asymgs" else ("PSNR", "SSIM", "LPIPS")
    for name in required:
        value = metrics.get(name)
        if name == "lpips" and value is None:
            value = metrics.get("lpips_vgg")
        require(isinstance(value, (int, float)) and math.isfinite(value), f"Missing/non-finite metric: {name}")
    return metrics


def evaluate(args, method, config):
    if not args.model:
        require(not args.data and not args.output, "--data/--output require --model. With no --model, test.sh performs the CUDA smoke check.")
        check(args, method, config)
        return
    require(not args.imports_only, "--imports-only cannot be combined with checkpoint evaluation.")
    require(args.data, "Checkpoint evaluation requires --data /path/to/dataset.")
    model = Path(args.model).expanduser().resolve()
    data = Path(args.data).expanduser().resolve()
    require(model.is_dir(), f"Checkpoint directory not found: {model}")
    require(data.is_dir(), f"Dataset directory not found: {data}")
    python = python_for(args, config)
    output = Path(args.output).expanduser().resolve() if args.output else method / "test_outputs" / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    require(not output.exists(), f"Output already exists: {output}. Choose a new directory; old results are never reused.")
    require(not output.is_relative_to(model) and not output.is_relative_to(data), "Evaluation output must be outside the original checkpoint and dataset directories.")
    iteration = args.iteration
    settings = None
    if method.name != "rl_asymgs":
        require((model / "cfg_args").is_file(), "Checkpoint requires cfg_args and point_cloud/iteration_N/point_cloud.ply.")
        settings = read_config(model / "cfg_args")
        if iteration == -1:
            candidates = [int(p.name.removeprefix("iteration_")) for p in (model / "point_cloud").glob("iteration_*") if p.name.removeprefix("iteration_").isdigit() and (p / "point_cloud.ply").is_file()]
            require(candidates, "No saved Gaussian point cloud found.")
            iteration = max(candidates)
        require((model / f"point_cloud/iteration_{iteration}/point_cloud.ply").is_file(), f"Missing point cloud at iteration {iteration}")
        images = args.images or settings.get("images", "images")
        require((data / images).is_dir(), f"Image directory not found: {data / images}")
        require((data / "sparse").is_dir() or (data / "transforms_test.json").is_file(), "Dataset requires COLMAP sparse reconstruction or transforms_test.json.")
        settings.update(source_path=str(data), model_path=str(output), images=images, eval=True)
        if args.resolution is not None:
            settings["resolution"] = args.resolution
    else:
        require((model / "nb-info.json").is_file(), "RL-AsymGS requires a NerfBaselines checkpoint directory containing nb-info.json.")
        require(args.images is None and args.resolution is None and args.iteration == -1, "AsymGS uses its saved NerfBaselines dataset settings; --images, --resolution and --iteration apply to the other backbones.")
    env = environment(method)
    if method.name == "rl_asymgs":
        env["NERFBASELINES_REGISTER"] = str(method / "evaluation_spec.py")
    if args.dry_run:
        print(f"Evaluation plan: {model} -> {output}; dataset={data}")
    else:
        check(args, method, config)
        output.mkdir(parents=True)
        (output / "run.json").write_text(json.dumps({"method": method.name, "checkpoint": str(model), "data": str(data), "iteration": iteration, "python": str(python), "status": "running"}, indent=2))
    try:
        if method.name == "rl_asymgs":
            cli = python.parent / "nerfbaselines"
            require(cli.is_file(), f"NerfBaselines CLI missing: {cli}")
            run([cli, "render", "--checkpoint", model, "--data", data, "--output", output / "predictions", "--split", "test", "--backend", "python"], method, env, args.dry_run)
            run([cli, "evaluate", output / "predictions", "--output", output / "results.json"], method, env, args.dry_run)
        else:
            if not args.dry_run:
                (output / "point_cloud").symlink_to(model / "point_cloud", target_is_directory=True)
                (output / "cfg_args").write_text(repr(argparse.Namespace(**settings)))
            run([python, "-B", method / "render.py", "-m", output, "--iteration", iteration, "--skip_train", "--quiet"], method, env, args.dry_run)
            if not args.dry_run:
                rendered = output / f"test/ours_{iteration}"
                renders = {p.name for p in (rendered / "renders").glob("*.png")}
                gts = {p.name for p in (rendered / "gt").glob("*.png")}
                require(renders and renders == gts, "No matched held-out render/ground-truth pairs were produced.")
            run([python, "-B", method / "metrics.py", "-m", output], method, env, args.dry_run)
        if not args.dry_run:
            metrics = validate_metrics(output / "results.json", method, iteration)
            record = json.loads((output / "run.json").read_text())
            record.update(status="passed", metrics=metrics)
            (output / "run.json").write_text(json.dumps(record, indent=2))
            print("Evaluation passed: " + str(output / "results.json"))
    except BaseException as error:
        if not args.dry_run:
            record = json.loads((output / "run.json").read_text())
            record.update(status="failed", error=str(error))
            (output / "run.json").write_text(json.dumps(record, indent=2))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("action", choices=("setup", "test", "_probe"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--env", help="Conda environment name (default: method-specific rlgs-* environment)")
    selection.add_argument("--prefix", help="Conda environment prefix")
    selection.add_argument("--python", help="Existing interpreter; setup requires --check-only")
    parser.add_argument("--check-only", action="store_true", help="For setup: check an existing environment without modifying it")
    parser.add_argument("--imports-only", action="store_true", help="Skip GPU kernel execution; this does not validate CUDA compatibility")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without creating environments or evaluation outputs")
    parser.add_argument("--jobs", type=int, default=4, help="CUDA build jobs")
    parser.add_argument("--model", help="Existing checkpoint directory; omitted: run a small CUDA smoke check")
    parser.add_argument("--data", help="Prepared local dataset directory")
    parser.add_argument("--output", help="New evaluation directory; never overwrites an existing directory")
    parser.add_argument("--iteration", type=int, default=-1, help="Saved iteration; default: latest point cloud")
    parser.add_argument("--images", help="Override saved image subdirectory (3DGS/TMGS/MCMC)")
    parser.add_argument("--resolution", type=int, help="Override saved image resolution (3DGS/TMGS/MCMC)")
    args = parser.parse_args()
    require(args.jobs > 0, "--jobs must be positive")
    method = ROOT / args.method
    config = json.loads((method / "runtime.json").read_text())
    if args.action == "setup":
        require(not args.model and not args.data and not args.output, "Checkpoint arguments belong to test.sh.")
        setup(args, method, config)
    elif args.action == "_probe":
        probe(args, method, config)
    else:
        evaluate(args, method, config)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
