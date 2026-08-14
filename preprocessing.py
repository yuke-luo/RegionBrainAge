#!/usr/bin/env python3
"""
Minimal T1w preprocessing pipeline (NIfTI input):
1) Input: T1w NIfTI (.nii / .nii.gz)
2) FSL reorientation to standard using fslreorient2std
3) Brain extraction (FreeSurfer SynthStrip) -> brain image
4) Affine registration to MNI152 T1 1mm brain template (FSL FLIRT, DOF=12) -> output in 1mm MNI space
5) Visual quality control: generate PNGs + index.html using FSL slicer + pngappend

Requirements:
- FreeSurfer 7.4.1 (SynthStrip)
- FSL (fslreorient2std, flirt, slicer, pngappend)
- Template (default): $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Dict, List, Optional

# -----------------------------
# Helpers
# -----------------------------

def strip_nii_ext(p: Path) -> str:
    """Return filename without .nii or .nii.gz"""
    n = p.name
    return n[:-7] if n.endswith(".nii.gz") else (n[:-4] if n.endswith(".nii") else p.stem)

def nifti_suffix(p: Path) -> str:
    """Return the NIfTI suffix to preserve for temporary aliases."""
    return ".nii.gz" if p.name.endswith(".nii.gz") else ".nii"

def safe_name_part(value: str) -> str:
    """Return a filesystem-friendly name component."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._-") or "part"

def compact_output_name(name: str, source: Path, max_len: int = 180) -> str:
    """Keep output directory names under common filesystem limits."""
    if len(name) <= max_len:
        return name
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
    return f"{name[: max_len - 12].rstrip('_')}__{digest}"

def relative_output_name(in_t1: Path, input_root: Path) -> str:
    """Create a collision-resistant output name from path relative to input_root."""
    try:
        rel = in_t1.relative_to(input_root)
    except ValueError:
        rel = Path(in_t1.name)

    parts = list(rel.parts)
    if parts:
        parts[-1] = strip_nii_ext(Path(parts[-1]))
    name = "__".join(safe_name_part(part) for part in parts)
    return compact_output_name(name, in_t1)

def unique_output_names(files: List[Path], input_path: Path, mode: str = "auto") -> List[str]:
    """Return stable, unique output names for input files.

    auto: use basenames unless duplicate basenames are detected, then use
    relative-path names. This preserves existing IXI behavior while making
    nested datasets such as ADNI safe.
    """
    if input_path.is_file():
        input_root = input_path.parent
    else:
        input_root = input_path

    basename_names = [safe_name_part(strip_nii_ext(path)) for path in files]
    has_duplicate_basenames = len(set(basename_names)) != len(basename_names)
    use_relative = mode == "relative" or (mode == "auto" and has_duplicate_basenames)
    if mode == "auto" and has_duplicate_basenames:
        print("Duplicate input basenames detected; using relative-path output names.")

    names = [
        relative_output_name(path, input_root) if use_relative else compact_output_name(name, path)
        for path, name in zip(files, basename_names)
    ]

    seen: Dict[str, int] = {}
    unique_names: List[str] = []
    for path, name in zip(files, names):
        if name in seen:
            digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
            name = compact_output_name(f"{name}__{digest}", path)
        seen[name] = seen.get(name, 0) + 1
        unique_names.append(name)

    return unique_names

def find_nifti_files(input_path: Path, recursive: bool = False) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    files: List[Path] = []
    if recursive:
        for root, dirnames, filenames in os.walk(input_path):
            dirnames[:] = [
                name
                for name in dirnames
                if not name.startswith(".")
                and name not in {"__pycache__", "node_modules"}
            ]
            root_path = Path(root)
            files.extend(
                root_path / name
                for name in filenames
                if name.endswith(".nii") or name.endswith(".nii.gz")
            )
    else:
        files.extend(sorted(input_path.glob("*.nii")))
        files.extend(sorted(input_path.glob("*.nii.gz")))

    return sorted(set(files))


def which_or_none(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def default_mni_template() -> Optional[Path]:
    fsldir = os.environ.get("FSLDIR", "")
    if not fsldir:
        return None
    p = Path(fsldir) / "data" / "standard" / "MNI152_T1_1mm_brain.nii.gz"
    return p if p.exists() else None

def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> bool:
    """Run command; return True if success, False if failed."""
    try:
        subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)
        return True
    except subprocess.CalledProcessError:
        return False

def find_fsl_cmd(name: str) -> Optional[str]:
    """Find an FSL binary either in PATH or in $FSLDIR/bin/."""
    if which_or_none(name):
        return name
    fsldir = os.environ.get("FSLDIR", "")
    if fsldir:
        candidate = Path(fsldir) / "bin" / name
        if candidate.exists():
            return str(candidate)
    return None

def find_synthstrip_cmd(explicit_cmd: Optional[str] = None) -> Optional[str]:
    """Find FreeSurfer SynthStrip in PATH, FreeSurfer env dirs, or nearby installs."""
    if explicit_cmd:
        expanded = Path(explicit_cmd).expanduser()
        if expanded.exists():
            resolved = expanded.resolve()
            if resolved.parent.name == "bin":
                os.environ.setdefault("FREESURFER_HOME", str(resolved.parent.parent))
            elif resolved.parent.name == "scripts" and resolved.parent.parent.name == "python":
                os.environ.setdefault("FREESURFER_HOME", str(resolved.parent.parent.parent))
            return str(resolved)
        if which_or_none(explicit_cmd):
            return explicit_cmd
        return None

    for cmd in ("mri_synthstrip", "synthstrip"):
        if which_or_none(cmd):
            return cmd

    freesurfer_roots: List[Path] = []
    for env_name in ("FREESURFER_HOME", "FS_HOME"):
        env_value = os.environ.get(env_name, "")
        if env_value:
            freesurfer_roots.append(Path(env_value).expanduser())

    fsldir = os.environ.get("FSLDIR", "")
    if fsldir:
        freesurfer_roots.append(Path(fsldir).expanduser().parent / "freesurfer")

    script_dir = Path(__file__).resolve().parent
    freesurfer_roots.extend(
        [
            script_dir.parent / "freesurfer",
            Path.cwd() / "freesurfer",
            Path.cwd().parent / "freesurfer",
        ]
    )

    seen = set()
    for root in freesurfer_roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        for rel in ("bin/mri_synthstrip", "python/scripts/mri_synthstrip", "bin/synthstrip"):
            candidate = root / rel
            if candidate.exists():
                os.environ.setdefault("FREESURFER_HOME", str(root))
                return str(candidate)

    return None

def safe_unlink(p: Path) -> None:
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass

def copy_or_symlink(src: Path, dst: Path) -> None:
    """Create a lightweight no-spaces alias for tools that mishandle spaced paths."""
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)

def prepare_work_input(in_t1: Path, work_dir: Path) -> Path:
    """Create a temporary input alias for tools that dislike spaced paths."""
    work_input = work_dir / f"input{nifti_suffix(in_t1)}"
    copy_or_symlink(in_t1, work_input)
    return work_input

# -----------------------------
# Pipeline steps
# -----------------------------
def run_fslreorient2std(in_t1: Path, out_t1: Path, reorient_cmd: str) -> bool:
    cmd = [reorient_cmd, str(in_t1), str(out_t1)]
    return run_cmd(cmd) and out_t1.exists()


def run_synthstrip(in_t1: Path, out_brain: Path, synthstrip_cmd: str) -> bool:
    # Common SynthStrip cmd: mri_synthstrip -i in.nii.gz -o brain.nii.gz
    cmd = [synthstrip_cmd, "-i", str(in_t1), "-o", str(out_brain)]
    return run_cmd(cmd) and out_brain.exists()

def run_flirt_affine(
    in_brain: Path,
    template: Path,
    out_brain_mni: Path,
    out_mat: Path,
    flirt_cmd: str,
) -> bool:
    cmd = [
        flirt_cmd,
        "-in", str(in_brain),
        "-ref", str(template),
        "-out", str(out_brain_mni),
        "-omat", str(out_mat),
        "-dof", "12",
    ]
    return run_cmd(cmd) and out_brain_mni.exists() and out_mat.exists()

def generate_qc(
    image: Path,
    out_dir: Path,
    overwrite: bool = False,
    width: int = 1500,
) -> Optional[Path]:
    """
    generate QC images and html file
    - slicer generates 9 views at x/y/z = 0.4/0.5/0.6
    - pngappend concatenates into one PNG
    """
    slicer_cmd = find_fsl_cmd("slicer")
    pngappend_cmd = find_fsl_cmd("pngappend")
    if slicer_cmd is None or pngappend_cmd is None:
        return None

    if out_dir.exists():
        if overwrite:
            shutil.rmtree(out_dir)
        else:
            idx = out_dir / "index.html"
            if idx.exists():
                return idx
            shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    index_html = out_dir / "index.html"
    with open(index_html, "w") as f:
        f.write('<HTML><TITLE>slicedir</TITLE><BODY BGCOLOR="#458dd1">\n')

    # temp slice files
    grota = out_dir / "grota.png"
    grotb = out_dir / "grotb.png"
    grotc = out_dir / "grotc.png"
    grotd = out_dir / "grotd.png"
    grote = out_dir / "grote.png"
    grotf = out_dir / "grotf.png"
    grotg = out_dir / "grotg.png"
    groth = out_dir / "groth.png"
    groti = out_dir / "groti.png"

    Q = strip_nii_ext(image)
    out_png = out_dir / f"{Q}.png"

    slicer_args = [
        slicer_cmd,
        str(image),
        "-s", "1",
        "-x", "0.4", str(grota),
        "-x", "0.5", str(grotb),
        "-x", "0.6", str(grotc),
        "-y", "0.4", str(grotd),
        "-y", "0.5", str(grote),
        "-y", "0.6", str(grotf),
        "-z", "0.4", str(grotg),
        "-z", "0.5", str(groth),
        "-z", "0.6", str(groti),
    ]
    if not run_cmd(slicer_args):
        return None

    pngappend_args = [
        pngappend_cmd,
        str(grota), "+", str(grotb), "+", str(grotc), "+",
        str(grotd), "+", str(grote), "+", str(grotf), "+",
        str(grotg), "+", str(groth), "+", str(groti),
        str(out_png),
    ]
    if not run_cmd(pngappend_args):
        return None

    # cleanup temp grots
    for p in (grota, grotb, grotc, grotd, grote, grotf, grotg, groth, groti):
        safe_unlink(p)

    with open(index_html, "a") as f:
        f.write(f'<a href="{out_png.name}"><img src="{out_png.name}" WIDTH={width} > {Q}</a><br>\n')
        f.write("</BODY></HTML>\n")

    return index_html if index_html.exists() else None





def preprocess_one(
    in_t1: Path,
    out_root: Path,
    template: Path,
    reorient_cmd: str,
    synthstrip_cmd: str,
    flirt_cmd: str,
    overwrite: bool = False,
    output_name: Optional[str] = None,
) -> Dict:
    name = output_name or safe_name_part(strip_nii_ext(in_t1))
    subdir = out_root / name

    if overwrite and subdir.exists():
        shutil.rmtree(subdir)
    subdir.mkdir(parents=True, exist_ok=True)

    final_brain_mni = subdir / f"{name}_brain_mni152_1mm.nii.gz"
    qc_dir = subdir / f"slicesdir_{name}"
    qc_index = qc_dir / "index.html"

    # Skip if already done
    if (not overwrite) and final_brain_mni.exists() and qc_index.exists():
        return {
            "input": str(in_t1),
            "status": "ok",
            "fail_stage": "",
            "skipped": True,
            "output_dir": str(subdir),
        }

    with tempfile.TemporaryDirectory(prefix="rba_preproc_") as work_tmp:
        work_dir = Path(work_tmp)
        tmp_reoriented = work_dir / "reoriented.nii.gz"
        tmp_brain = work_dir / "brain.nii.gz"
        tmp_mat = work_dir / "affine.mat"
        tmp_final = work_dir / f"{name}_brain_mni152_1mm.nii.gz"
        tmp_qc_dir = work_dir / "slicesdir"

        work_input = prepare_work_input(in_t1, work_dir)

        # Reorientation to standard
        if not run_fslreorient2std(work_input, tmp_reoriented, reorient_cmd):
            print("reorientation failed - skipping this image: {}".format(in_t1))
            return {
                "input": str(in_t1),
                "status": "failed",
                "fail_stage": "reorient",
                "skipped": False,
                "output_dir": str(subdir),
            }

        # Skull stripping
        if not run_synthstrip(tmp_reoriented, tmp_brain, synthstrip_cmd):
            print("skull-stripping failed - skipping this image: {}".format(in_t1))
            return {
                "input": str(in_t1),
                "status": "failed",
                "fail_stage": "skull_stripping",
                "skipped": False,
                "output_dir": str(subdir),
            }

        # Registration
        if not run_flirt_affine(tmp_brain, template, tmp_final, tmp_mat, flirt_cmd):
            print("registration failed - skipping this image: {}".format(in_t1))
            return {
                "input": str(in_t1),
                "status": "failed",
                "fail_stage": "registration",
                "skipped": False,
                "output_dir": str(subdir),
            }

        # QC PNG + HTML
        idx = generate_qc(tmp_final, tmp_qc_dir, overwrite=True, width=1500)
        if idx is None:
            print("QC slices failed - skipping this image: {}".format(in_t1))
            return {
                "input": str(in_t1),
                "status": "failed",
                "fail_stage": "qc_slices",
                "skipped": False,
                "output_dir": str(subdir),
            }

        shutil.copy2(tmp_final, final_brain_mni)
        if qc_dir.exists():
            shutil.rmtree(qc_dir)
        shutil.copytree(tmp_qc_dir, qc_dir)

    return {
        "input": str(in_t1),
        "status": "ok",
        "fail_stage": "",
        "skipped": False,
        "output_dir": str(subdir),
    }



def write_summary_csv(records: List[Dict], out_csv: Path) -> None:
    fields = ["input", "status", "fail_stage", "skipped", "output_dir"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fields})


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal T1w preprocessing: SynthStrip + FLIRT + HTML QC (slicesdir)")
    parser.add_argument("-i", "--input", required=True, type=str, help="Input .nii/.nii.gz OR a folder containing them")
    parser.add_argument("-o", "--out", required=True, type=str, help="Output folder")
    parser.add_argument("--recursive", action="store_true", help="Search for NIfTI files recursively in folders")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of subjects to preprocess in parallel (default: 1)",
    )
    parser.add_argument(
        "--synthstrip-cmd",
        type=str,
        default=None,
        help="Path or command name for FreeSurfer SynthStrip (mri_synthstrip/synthstrip)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument(
        "--name-mode",
        choices=("auto", "basename", "relative"),
        default="auto",
        help=(
            "How to name output folders. 'auto' keeps basenames unless duplicate "
            "basenames are detected, then uses relative paths (default)."
        ),
    )
    args = parser.parse_args()

    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1.")

    input_path = Path(args.input).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Template
    template = default_mni_template()
    if template is None:
        raise SystemExit(
            "Could not find default MNI template. Set $FSLDIR so this exists: "
            "$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz"
        )
    if not template.exists():
        raise SystemExit(f"Template not found: {template}")

    # synthstrip cmd
    synthstrip_cmd = find_synthstrip_cmd(args.synthstrip_cmd)
    if synthstrip_cmd is None:
        raise SystemExit(
            "Could not find SynthStrip command. Load FreeSurfer, set $FREESURFER_HOME, "
            "or pass --synthstrip-cmd /path/to/mri_synthstrip."
        )

    reorient_cmd = find_fsl_cmd("fslreorient2std")
    if reorient_cmd is None:
        raise SystemExit("Could not find 'fslreorient2std' in PATH or $FSLDIR/bin.")

    # required FSL tools
    flirt_cmd = find_fsl_cmd("flirt")
    if flirt_cmd is None:
        raise SystemExit("Could not find 'flirt' in PATH or $FSLDIR/bin.")
    if find_fsl_cmd("slicer") is None or find_fsl_cmd("pngappend") is None:
        raise SystemExit("Could not find FSL 'slicer' and/or 'pngappend'.")

    # inputs
    files = find_nifti_files(input_path, recursive=args.recursive)
    if not files:
        raise SystemExit(f"No .nii/.nii.gz files found in: {input_path}")
    output_names = unique_output_names(files, input_path=input_path, mode=args.name_mode)

    records: List[Dict] = []
    max_workers = min(args.jobs, len(files))
    if max_workers == 1:
        for f, output_name in zip(files, output_names):
            print(f"\nProcessing: {f}")
            rec = preprocess_one(
                in_t1=f,
                out_root=out_root,
                template=template,
                reorient_cmd=reorient_cmd,
                synthstrip_cmd=synthstrip_cmd,
                flirt_cmd=flirt_cmd,
                overwrite=args.overwrite,
                output_name=output_name,
            )
            records.append(rec)
    else:
        print(f"Processing {len(files)} subjects with {max_workers} parallel workers.")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(
                preprocess_one,
                files,
                repeat(out_root),
                repeat(template),
                repeat(reorient_cmd),
                repeat(synthstrip_cmd),
                repeat(flirt_cmd),
                repeat(args.overwrite),
                output_names,
            )
            records.extend(results)

    summary_csv = out_root / "preprocess_summary.csv"
    write_summary_csv(records, summary_csv)
    print(f"\nSummary saved to: {summary_csv}")


if __name__ == "__main__":
    main()
