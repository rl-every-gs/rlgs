#!/usr/bin/env python3
"""Orchestrate the frozen TNT recipe without changing training source or defaults."""

import argparse
import contextlib
import csv
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parent.parent
LOCK_DIR = ROOT / "reproducibility" / "tnt_camera_ready_v1"
METHOD = "ours_30000"


def log(message):
    print(f"[{datetime.datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    # All report paths are inside a newly reserved run directory.
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def verify_sources():
    entries = json.loads((LOCK_DIR / "sources.json").read_text())
    for name, expected in entries.items():
        path = ROOT / name
        require(path.is_file() and sha256(path) == expected,
                f"Frozen source changed or missing: {name}. Restore v1 before running.")
    log(f"Frozen source verified: {len(entries)} files (including local CUDA extension sources).")


def verify_data(data_root, scene, recipe):
    scene_root = data_root / scene
    expected = recipe["data"]["scenes"][scene]
    images = scene_root / "images"
    require(images.is_dir(), f"Missing dataset: {images}")
    actual = sorted(path.name for path in images.iterdir() if path.is_file())
    require(len(actual) == expected["image_count"], f"Image count differs from the validated dataset: {scene}")
    names_digest = hashlib.sha256("".join(f"{name}\n" for name in actual).encode()).hexdigest()
    require(names_digest == expected["image_names_sha256"],
            f"Image names differ from the validated dataset: {scene}")
    for name, digest in expected["sha256"].items():
        path = scene_root / name
        require(path.is_file() and sha256(path) == digest,
                f"Dataset checksum mismatch: {path}. Existing data will not be overwritten.")
    # The loader prefers a cached PLY over points3D.bin. Detect a stale/replaced cache.
    cached = scene_root / "sparse/0/points3D.ply"
    if cached.exists():
        require(sha256(cached) == expected["cached_ply_sha256"],
                f"Unrecognized cached initialization: {cached}. Use a fresh data root.")
    return {"images": len(actual), "train_views": expected["train_views"],
            "test_views": expected["test_views"], "checksums_verified": True}


@contextlib.contextmanager
def exclusive_lock(path):
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another reproduction is using {path}") from exc
        yield


def prepare_data(args, scenes, recipe):
    args.data_root.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(args.data_root.parent / ".rlgs-tnt-data.lock"):
        missing = []
        for scene in scenes:
            if (args.data_root / scene).exists():
                verify_data(args.data_root, scene, recipe)
                log(f"Reusing verified dataset: {args.data_root / scene}")
            else:
                missing.append(scene)
        if not missing:
            return
        archive = args.archive
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            require(shutil.which("curl"), "curl is required to download TNT.")
            partial = archive.with_name(archive.name + ".part")
            log(f"Downloading official 3DGS TNT/DB archive ({recipe['data']['archive_bytes']} bytes).")
            subprocess.run([
                "curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "20",
                "--speed-limit", "1024", "--speed-time", "60", "--continue-at", "-",
                "--output", str(partial), args.download_url,
            ], check=True)
            verify_archive(partial, recipe)
            partial.rename(archive)
        else:
            verify_archive(archive, recipe)
        args.data_root.mkdir(parents=True, exist_ok=True)
        # Extract only TNT, under a temporary directory on the destination filesystem.
        # Existing scene directories are never repaired, merged, or replaced.
        with tempfile.TemporaryDirectory(prefix=".rlgs-tnt-", dir=args.data_root) as stage:
            stage = Path(stage)
            with zipfile.ZipFile(archive) as bundle:
                selected = []
                for member in bundle.infolist():
                    name = PurePosixPath(member.filename)
                    require(not name.is_absolute() and ".." not in name.parts and "\\" not in member.filename,
                            f"Unsafe ZIP path: {member.filename}")
                    require(not stat.S_ISLNK(member.external_attr >> 16), "ZIP symlinks are not supported.")
                    if len(name.parts) >= 2 and name.parts[0] == "tandt" and name.parts[1] in missing:
                        selected.append(member)
                require(selected, "Archive does not contain the expected tandt/train or tandt/truck layout.")
                needed = sum(member.file_size for member in selected)
                require(shutil.disk_usage(stage).free > needed + 64 * 1024**2,
                        "Insufficient free disk space to extract TNT.")
                for member in selected:
                    bundle.extract(member, stage)  # zipfile also verifies each extracted member's CRC.
            for scene in missing:
                verify_data(stage / "tandt", scene, recipe)
            for scene in missing:
                destination = args.data_root / scene
                require(not destination.exists(), f"Dataset appeared during extraction: {destination}")
                (stage / "tandt" / scene).rename(destination)
                log(f"Installed and verified: {destination}")


def verify_archive(path, recipe):
    require(path.stat().st_size == recipe["data"]["archive_bytes"], f"Incomplete archive: {path}")
    require(sha256(path) == recipe["data"]["archive_sha256"],
            f"Official archive checksum mismatch: {path}. Keep the file for inspection; it will not be overwritten.")


def verify_runtime(recipe):
    expected = recipe["runtime"]
    require(sys.version_info[:2] == (3, 10), f"Expected Python 3.10, got {sys.version.split()[0]} at {sys.executable}")
    for package, version in expected["packages"].items():
        actual = importlib.metadata.version(package)
        require(actual == version, f"Expected {package}=={version}, got {actual} in {sys.executable}")
    # Imports check the extension ABI; no training or CUDA kernels run in --check-only.
    import torch
    import torchvision
    import diff_gaussian_rasterization
    import diff_gaussian_rasterization._C
    import simple_knn._C
    import fused_ssim
    import fused_ssim_cuda
    require(torch.version.cuda == expected["cuda_runtime"], f"Unexpected CUDA runtime: {torch.version.cuda}")
    require(torch.cuda.is_available(), "PyTorch cannot access a CUDA GPU.")
    require("pixel_weights" in diff_gaussian_rasterization.GaussianRasterizationSettings._fields,
            "Wrong rasterizer installed: build this repository's customized CUDA extension.")
    for name in ("helper_rl", "helper_tmgs", "helper_data", "helper_optimzer", "lpipsPyTorch"):
        __import__(name)
    details = {
        "python_executable": sys.executable, "python": sys.version,
        "torch": str(torch.__version__), "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "MKL_THREADING_LAYER": os.environ["MKL_THREADING_LAYER"],
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "extensions": {"rasterizer": diff_gaussian_rasterization._C.__file__,
                       "knn": simple_knn._C.__file__, "ssim": fused_ssim_cuda.__file__},
    }
    log(f"Runtime verified: {sys.executable}; torch {torch.__version__}; CUDA {torch.version.cuda}; {details['gpu']}")
    return details


def commands(scene, args, recipe):
    output = args.output_root / scene
    training = [sys.executable, "-u", recipe["trainer"],
                "-s", str(args.data_root / scene), "-m", str(output),
                "--budget", str(recipe["scenes"][scene]["budget"]), *recipe["training_args"]]
    render = [sys.executable, "-u", "render.py", "-m", str(output),
              "--iteration", "30000", "--skip_train", "--quiet"]
    metrics = [sys.executable, "-u", "metrics.py", "-m", str(output)]
    return {"train": training, "render": render, "metrics": metrics}


def run_step(name, command, output):
    log(f"{output.name}: starting {name}; log: {output / (name + '.log')}")
    started = time.monotonic()
    with (output / f"{name}.log").open("x") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    write_json(output / f"{name}_status.json", {"exit_code": completed.returncode, "wall_seconds": elapsed})
    require(completed.returncode == 0, f"{name} failed with exit {completed.returncode}; see {output / (name + '.log')}")
    log(f"{output.name}: {name} finished in {elapsed:.1f}s")
    return elapsed


def validate_model(output, scene, recipe):
    final = output / "point_cloud/iteration_30000"
    for name in ("point_cloud.ply", "grd_policy_network.torch", "densi_policy_network.torch"):
        path = final / name
        require(path.is_file() and path.stat().st_size > 0, f"Missing final artifact: {path}")
    with (final / "point_cloud.ply").open("rb") as handle:
        header = handle.read(16384).split(b"end_header\n", 1)[0]
    match = re.search(rb"(?m)^element vertex (\d+)\r?$", header)
    count = int(match[1]) if match else None
    require(count == recipe["scenes"][scene]["budget"], f"{scene}: final count {count} differs from paper budget.")
    result = (output / "result.txt").read_text()
    psnr = re.findall(r"(?m)^test-psnr_([-+0-9.eE]+)-iteration_30000$", result)
    require(psnr and math.isfinite(float(psnr[-1])), f"{scene}: no finite 30k test PSNR in result.txt")
    return {"gaussians": count, "training_test_psnr": float(psnr[-1])}


def validate_renders(output, scene, recipe):
    expected = recipe["data"]["scenes"][scene]["test_views"]
    expected_names = {f"{index:05d}.png" for index in range(expected)}
    for folder in ("renders", "gt"):
        path = output / "test" / METHOD / folder
        require(path.is_dir(), f"Missing render directory: {path}")
        actual = {entry.name for entry in path.iterdir() if entry.is_file() and entry.stat().st_size > 0}
        require(actual == expected_names, f"{scene}: incomplete {folder}; expected {expected} images.")


def validate_metrics(output, scene, recipe):
    # metrics.py catches exceptions internally, so exit code 0 alone is insufficient.
    validate_renders(output, scene, recipe)
    metrics = json.loads((output / "results.json").read_text())[METHOD]
    per_view = json.loads((output / "per_view.json").read_text())[METHOD]
    expected_names = {path.name for path in (output / "test" / METHOD / "renders").iterdir()}
    for key in ("PSNR", "SSIM", "LPIPS"):
        value = metrics[key]
        require(isinstance(value, (int, float)) and math.isfinite(value), f"Non-finite {key}: {output}")
        require(set(per_view[key]) == expected_names, f"Incomplete per-view {key}: {output}")
        require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in per_view[key].values()),
                f"Non-finite per-view {key}: {output}")
        require(math.isclose(sum(per_view[key].values()) / len(expected_names), value, abs_tol=1e-5),
                f"Mean {key} disagrees with per-view results: {output}")
    require(metrics["PSNR"] > 0 and 0 <= metrics["SSIM"] <= 1 and metrics["LPIPS"] >= 0,
            f"Out-of-range metrics: {output}")
    return {key: metrics[key] for key in ("PSNR", "SSIM", "LPIPS")}


def summarize(output, scene, recipe):
    result = {"scene": scene, **validate_model(output, scene, recipe),
              **validate_metrics(output, scene, recipe)}
    reference = recipe["scenes"][scene]["reference_metrics"]
    result["reference_metrics"] = reference
    result["PSNR_delta_from_reference"] = result["PSNR"] - reference["PSNR"]
    # Report the observed difference; a previous single run is not an accuracy guarantee.
    log(f"{scene}: PSNR={result['PSNR']:.5f}, SSIM={result['SSIM']:.6f}, "
        f"LPIPS={result['LPIPS']:.6f}, Gaussians={result['gaussians']}, "
        f"PSNR delta={result['PSNR_delta_from_reference']:+.5f} dB")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples (from any directory):
  bash /path/to/repo/reproduce_tnt.sh --dry-run
  bash /path/to/repo/reproduce_tnt.sh --check-only
  bash /path/to/repo/reproduce_tnt.sh
  bash /path/to/repo/reproduce_tnt.sh --scene train --output-root /new/output
  bash /path/to/repo/reproduce_tnt.sh --prepare-data-only --data-root /new/tandt

The default run downloads missing data, trains each scene once, renders held-out
views, evaluates PSNR/SSIM/LPIPS, and writes summary.json/summary.csv.
Existing output roots and inconsistent datasets are never overwritten.
Install the pinned method environment from the method directory:
  bash setup.sh
  conda activate rlgs-tmgs
Build prerequisites: Conda, an NVIDIA driver, and a C++ compiler (g++).
The setup script installs the pinned CUDA Toolkit and builds local extensions.
PYTHON selects an explicit interpreter. DATA_ROOT and OUTPUT_ROOT override paths.
Use --expected-host <hostname -s> only when a strict host-name guard is desired.
""")
    parser.add_argument("--scene", choices=("all", "train", "truck"), default="all")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", "/home/stack/data/tandt")))
    parser.add_argument("--output-root", type=Path,
                        default=Path(os.environ.get("OUTPUT_ROOT", str(ROOT / "output/rlgs_tmgs_tnt_cr_v1"))))
    parser.add_argument("--archive", type=Path, help="Official tandt_db.zip cache (also supports an existing local ZIP).")
    parser.add_argument("--download-url", help="Mirror URL; bytes must match the frozen official archive SHA256.")
    parser.add_argument("--expected-host", help="Optional required hostname -s; defaults to the current host.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Verify source/existing data and print commands; no writes or GPU checks.")
    mode.add_argument("--check-only", action="store_true", help="Verify environment, source and existing data; no downloads or training.")
    mode.add_argument("--prepare-data-only", action="store_true", help="Download/extract/verify missing data, then exit; no GPU checks.")
    mode.add_argument("--validate-results", action="store_true", help="Read and validate completed outputs; no writes, rendering or training.")
    return parser.parse_args()


def main():
    args = parse_args()
    # Also enforce the environment when the Python entry is used directly.
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    actual_host = os.uname().nodename.split(".")[0]
    if args.expected_host:
        require(actual_host == args.expected_host,
                f"Wrong host: expected {args.expected_host}, got {os.uname().nodename}. No work started.")
    else:
        args.expected_host = actual_host
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.archive = (args.archive or args.data_root.parent / ".cache/rlgs-tnt-cr-v1/tandt_db.zip").expanduser().resolve()
    recipe = json.loads((LOCK_DIR / "recipe.json").read_text())
    args.download_url = args.download_url or recipe["data"]["download_url"]
    scenes = ("train", "truck") if args.scene == "all" else (args.scene,)
    log(f"Recipe: {recipe['release']}; host: {args.expected_host}; interpreter: {sys.executable}")
    verify_sources()
    if args.validate_results:
        for scene in scenes:
            summarize(args.output_root / scene, scene, recipe)
        return
    if args.dry_run or args.check_only:
        if args.check_only:
            verify_runtime(recipe)
        for scene in scenes:
            if (args.data_root / scene).exists():
                verify_data(args.data_root, scene, recipe)
                log(f"Verified existing dataset: {scene}")
            else:
                require(not args.check_only, f"Missing {scene}; run --prepare-data-only first.")
                log(f"Would download {args.download_url} and install {args.data_root / scene}")
            for name, command in commands(scene, args, recipe).items():
                print(f"{name}: {shlex.join(command)}")
        log("Checks complete; no training, rendering, metrics, downloads or output writes performed.")
        return
    if args.prepare_data_only:
        prepare_data(args, scenes, recipe)
        return
    require(not args.output_root.exists(), f"Refusing to overwrite existing output root: {args.output_root}")
    runtime = verify_runtime(recipe)
    prepare_data(args, scenes, recipe)
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    # One runner per GPU/interpreter/host; prevents duplicate launches across output roots.
    gpu_id = hashlib.sha256((args.expected_host + str(Path(sys.executable).resolve()) +
                            str(os.environ.get("CUDA_VISIBLE_DEVICES", "default"))).encode()).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"rlgs-tnt-{os.getuid()}-{gpu_id}.lock"
    with exclusive_lock(lock_path):
        args.output_root.mkdir()  # Atomic reservation; fail even if another caller created an empty directory.
        try:
            write_json(args.output_root / "runtime.json", runtime)
            for name in ("recipe.json", "sources.json"):
                shutil.copyfile(LOCK_DIR / name, args.output_root / name)
            subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], check=True,
                           stdout=(args.output_root / "environment.freeze.txt").open("x"))
            # Download/cache LPIPS weights before the expensive training steps.
            log("Preparing VGG16/LPIPS evaluation weights (downloads once if absent).")
            from lpipsPyTorch.modules.lpips import LPIPS
            lpips = LPIPS(net_type="vgg").eval()
            del lpips
            results = []
            for scene in scenes:
                output = args.output_root / scene
                output.mkdir()
                steps = commands(scene, args, recipe)
                write_json(output / "commands.json", steps)
                write_json(output / "data_check.json", verify_data(args.data_root, scene, recipe))
                with (output / "commands.sh").open("x") as handle:
                    handle.write("#!/usr/bin/env bash\nset -Eeuo pipefail\n")
                    handle.write(f"cd -- {shlex.quote(str(ROOT))}\n")
                    handle.write("export MKL_THREADING_LAYER=GNU CUBLAS_WORKSPACE_CONFIG=:4096:8\n")
                    for command in steps.values():
                        handle.write(shlex.join(command) + "\n")
                times = {"train": run_step("train", steps["train"], output)}
                validate_model(output, scene, recipe)
                times["render"] = run_step("render", steps["render"], output)
                validate_renders(output, scene, recipe)
                times["metrics"] = run_step("metrics", steps["metrics"], output)
                result = summarize(output, scene, recipe)
                result["wall_seconds"] = times
                write_json(output / "validated_result.json", result)
                results.append(result)
            write_json(args.output_root / "summary.json", {"release": recipe["release"], "scenes": results})
            with (args.output_root / "summary.csv").open("x", newline="") as handle:
                fields = ("scene", "gaussians", "PSNR", "SSIM", "LPIPS", "PSNR_delta_from_reference")
                writer = csv.DictWriter(handle, fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(results)
            write_json(args.output_root / "SUCCESS.json", {"completed_at": datetime.datetime.now().astimezone().isoformat()})
            log(f"All requested scenes completed and validated. Summary: {args.output_root / 'summary.csv'}")
        except BaseException as exc:
            write_json(args.output_root / "FAILED.json", {"error": str(exc), "type": type(exc).__name__})
            raise


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        log(f"FAILED: {error}")
        sys.exit(1)
    except KeyboardInterrupt:
        log("Interrupted; partial outputs were preserved. Use a new output root for another run.")
        sys.exit(130)
