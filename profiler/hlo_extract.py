"""Extract and visualise XLA HLO graphs (before + after optimisation passes).

Pipeline
--------
1.  Run the experiment via ``_hlo_wrapper.py`` which monkey-patches
    ``xm.mark_step`` to capture the *pre-optimisation* HLO text using
    ``torch_xla._XLAC._get_xla_tensors_hlo``.
2.  The Neuron runtime writes *post-optimisation* ``.hlo_module.pb`` protobuf
    files to the compile-cache directory.
3.  Parse both representations into ``HloModule`` data-structures.
4.  Render each as a colour-coded Graphviz PNG.

Output (saved under ``<experiment>/artifacts/<impl>/``):
    hlo_before_optimizations.png   – raw graph from torch_xla
    hlo_after_optimizations.png    – optimised HLO sent to neuron-cc
    hlo_before_optimizations.txt   – raw HLO text
    hlo_after_optimizations.txt    – reconstructed HLO text from protobuf
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import runner

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HloInstruction:
    name: str
    shape: str
    opcode: str
    operands: list[str] = field(default_factory=list)
    is_root: bool = False


@dataclass
class HloComputation:
    name: str
    instructions: list[HloInstruction] = field(default_factory=list)
    is_entry: bool = False


@dataclass
class HloModule:
    name: str
    entry: HloComputation | None = None
    computations: dict[str, HloComputation] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Op-category styling for DOT nodes   (fillcolor, dot-shape)
# ---------------------------------------------------------------------------

_ARITH  = ("#C5E1A5", "ellipse")
_MATMUL = ("#FFB74D", "box")
_LAYOUT = ("#FFF9C4", "box")
_REDUCE = ("#F8BBD0", "ellipse")
_GATHER = ("#FFD54F", "ellipse")

OP_STYLES: dict[str, tuple[str, str]] = {
    "parameter":            ("#B3D9FF", "box"),
    "constant":             ("#E0E0E0", "plaintext"),
    "iota":                 ("#E0E0E0", "plaintext"),
    "add": _ARITH, "subtract": _ARITH, "multiply": _ARITH,
    "divide": _ARITH, "negate": _ARITH, "abs": _ARITH,
    "maximum": _ARITH, "minimum": _ARITH, "clamp": _ARITH,
    "compare": _ARITH, "select": _ARITH, "and": _ARITH, "or": _ARITH,
    "not": _ARITH,
    "exp": _ARITH, "log": _ARITH, "sqrt": _ARITH, "rsqrt": _ARITH,
    "tanh": _ARITH, "convert": _ARITH, "bitcast-convert": _ARITH,
    "power": _ARITH, "remainder": _ARITH, "floor": _ARITH, "ceil": _ARITH,
    "dot": _MATMUL, "convolution": _MATMUL,
    "reshape": _LAYOUT, "broadcast": _LAYOUT, "transpose": _LAYOUT,
    "bitcast": _LAYOUT, "slice": _LAYOUT, "pad": _LAYOUT,
    "concatenate": _LAYOUT, "reverse": _LAYOUT, "copy": _LAYOUT,
    "reduce": _REDUCE, "reduce-window": _REDUCE, "all-reduce": _REDUCE,
    "gather": _GATHER, "scatter": _GATHER,
    "dynamic-slice": _GATHER, "dynamic-update-slice": _GATHER,
    "custom-call":          ("#EF9A9A", "octagon"),
    "fusion":               ("#CE93D8", "doubleoctagon"),
    "while":                ("#80DEEA", "diamond"),
    "conditional":          ("#80DEEA", "diamond"),
    "call":                 ("#80DEEA", "diamond"),
    "tuple":                ("#F5F5F5", "box"),
    "get-tuple-element":    ("#F5F5F5", "box"),
}
_DEFAULT_STYLE = ("#FFFFFF", "ellipse")

# ---------------------------------------------------------------------------
# Primitive-type id → short name  (XLA PrimitiveType enum)
# ---------------------------------------------------------------------------
_PTYPE = {
    0: "invalid", 1: "pred",
    2: "s8", 3: "s16", 4: "s32", 5: "s64",
    6: "u8", 7: "u16", 8: "u32", 9: "u64",
    10: "f16", 11: "f32", 12: "f64",
    15: "c64", 16: "bf16", 17: "token", 18: "c128",
    20: "f8e5m2", 23: "f8e4m3fn",
}

# ---------------------------------------------------------------------------
# HLO text parsing  (for pre-opt dumps from _get_xla_tensors_hlo)
# ---------------------------------------------------------------------------

def parse_hlo_text(text: str) -> HloModule:
    m = re.match(r"HloModule\s+(\S+)", text)
    mod_name = m.group(1).rstrip(",") if m else "unknown"
    module = HloModule(name=mod_name)

    lines = text.split("\n")
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        is_entry = stripped.startswith("ENTRY")
        header = re.match(r"(?:ENTRY\s+)?%(\S+)\s*\(", stripped)
        if header and "{" in lines[i]:
            cname = header.group(1)
            comp = HloComputation(name=cname, is_entry=is_entry)
            depth = lines[i].count("{") - lines[i].count("}")
            i += 1
            while i < len(lines) and depth > 0:
                ln = lines[i].strip()
                depth += ln.count("{") - ln.count("}")
                if ln and ln != "}" and not ln.startswith("//"):
                    inst = _parse_text_instruction(ln)
                    if inst:
                        comp.instructions.append(inst)
                i += 1
            module.computations[cname] = comp
            if is_entry:
                module.entry = comp
            continue
        i += 1
    return module


def _parse_text_instruction(line: str) -> HloInstruction | None:
    is_root = line.startswith("ROOT")
    if is_root:
        line = line[4:].strip()
    m = re.match(r"%(\S+)\s*=\s*", line)
    if not m:
        return None
    name = m.group(1)
    rest = line[m.end():]
    m2 = re.search(r"(\b[a-zA-Z][\w.-]*)\(", rest)
    if not m2:
        return None
    opcode = m2.group(1)
    shape_raw = rest[:m2.start()].strip()
    shape = re.sub(r"\{[0-9,E]*\}", "", shape_raw)
    after = rest[m2.end():]
    depth, idx = 1, 0
    while idx < len(after) and depth > 0:
        ch = after[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        idx += 1
    ops_str = after[:max(idx - 1, 0)]
    operands = re.findall(r"%([a-zA-Z0-9_.]+)", ops_str)
    return HloInstruction(name=name, shape=shape, opcode=opcode,
                          operands=operands, is_root=is_root)


# ---------------------------------------------------------------------------
# HLO protobuf parsing  (for post-opt .pb from Neuron compile cache)
# ---------------------------------------------------------------------------

def _shape_str(shape_proto) -> str:
    """Convert a ShapeProto to a compact string like ``f32[256,1024]``."""
    etype = int(shape_proto.element_type)
    tname = _PTYPE.get(etype, f"t{etype}")
    dims = list(shape_proto.dimensions)
    if etype == 13:  # TUPLE
        subs = [_shape_str(s) for s in shape_proto.tuple_shapes]
        return "(" + ", ".join(subs) + ")"
    if dims:
        return f"{tname}[{','.join(str(d) for d in dims)}]"
    return tname


def parse_hlo_pb(pb_path: Path) -> HloModule:
    """Parse a Neuron ``model.hlo_module.pb`` into an :class:`HloModule`."""
    from torch_neuronx.pyhlo.service import hlo_pb2

    data = pb_path.read_bytes()
    proto = hlo_pb2.HloModuleProto()
    proto.ParseFromString(data)

    module = HloModule(name=proto.name)

    for comp_proto in proto.computations:
        is_entry = comp_proto.name == proto.entry_computation_name
        comp = HloComputation(name=comp_proto.name, is_entry=is_entry)

        id_to_name: dict[int, str] = {}
        for inst in comp_proto.instructions:
            id_to_name[inst.id] = inst.name

        for inst in comp_proto.instructions:
            operands = [id_to_name[oid] for oid in inst.operand_ids if oid in id_to_name]
            is_root = inst.id == comp_proto.root_id
            comp.instructions.append(HloInstruction(
                name=inst.name,
                shape=_shape_str(inst.shape),
                opcode=inst.opcode,
                operands=operands,
                is_root=is_root,
            ))

        module.computations[comp.name] = comp
        if is_entry:
            module.entry = comp

    return module


def hlo_module_to_text(module: HloModule) -> str:
    """Reconstruct an approximate HLO text from an :class:`HloModule`."""
    lines = [f"HloModule {module.name}"]
    for comp in module.computations.values():
        prefix = "ENTRY " if comp.is_entry else ""
        lines.append(f"\n{prefix}%{comp.name} {{")
        for inst in comp.instructions:
            root = "ROOT " if inst.is_root else ""
            ops = ", ".join(f"%{o}" for o in inst.operands)
            lines.append(f"  {root}%{inst.name} = {inst.shape} {inst.opcode}({ops})")
        lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graph coarsening
# ---------------------------------------------------------------------------

def _coarsen(instructions: list[HloInstruction], target: int) -> list[HloInstruction]:
    if not instructions or len(instructions) <= target:
        return instructions

    groups: list[list[HloInstruction]] = [[instructions[0]]]
    for inst in instructions[1:]:
        if inst.opcode == groups[-1][0].opcode and not inst.is_root:
            groups[-1].append(inst)
        else:
            groups.append([inst])

    name_map: dict[str, str] = {}
    result: list[HloInstruction] = []
    for grp in groups:
        rep = grp[0].name
        for g in grp:
            name_map[g.name] = rep
        if len(grp) == 1:
            result.append(grp[0])
        else:
            names_in = {g.name for g in grp}
            ext_ops = []
            for g in grp:
                ext_ops.extend(o for o in g.operands if o not in names_in)
            result.append(HloInstruction(
                name=rep, shape=grp[0].shape,
                opcode=f"{grp[0].opcode} x{len(grp)}",
                operands=list(dict.fromkeys(ext_ops)),
                is_root=any(g.is_root for g in grp),
            ))

    for inst in result:
        inst.operands = list(dict.fromkeys(name_map.get(o, o) for o in inst.operands))
    return result


# ---------------------------------------------------------------------------
# DOT generation
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("<", "\\<").replace(">", "\\>")


def _nid(name: str) -> str:
    return "n_" + re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _short_shape(shape: str, limit: int = 48) -> str:
    return shape if len(shape) <= limit else shape[:limit] + "…"


def hlo_to_dot(
    module: HloModule,
    title: str = "",
    max_nodes: int = 250,
) -> str:
    comp = module.entry
    if comp is None:
        return 'digraph G { label="No ENTRY computation found" }'

    orig_n = len(comp.instructions)
    insts = comp.instructions
    coarsened = orig_n > max_nodes
    if coarsened:
        insts = _coarsen(insts, max_nodes)

    op_counts: dict[str, int] = {}
    for inst in comp.instructions:
        op_counts[inst.opcode] = op_counts.get(inst.opcode, 0) + 1
    summary = "\\l".join(
        f"{op}: {c}" for op, c in sorted(op_counts.items(), key=lambda x: -x[1])[:20]
    ) + "\\l"

    lbl = f"{_esc(title)}\\nModule: {_esc(module.name)}\\n{orig_n} instructions"
    if coarsened:
        lbl += f"  (coarsened to {len(insts)} nodes)"

    out = [
        "digraph G {",
        "  rankdir=TB;",
        "  concentrate=true;",
        f'  label="{lbl}";',
        "  labelloc=t; fontsize=16;",
        '  node [style=filled, fontsize=9, fontname="Helvetica"];',
        '  edge [arrowsize=0.4, color="#666666"];',
    ]

    names = {i.name for i in insts}
    for inst in insts:
        base_op = inst.opcode.split(" x")[0]
        fill, nshape = OP_STYLES.get(base_op, _DEFAULT_STYLE)
        parts = [_esc(inst.opcode)]
        if inst.shape:
            parts.append(_esc(_short_shape(inst.shape)))
        label = "\\n".join(parts)
        pen = ", penwidth=3" if inst.is_root else ""
        out.append(
            f'  {_nid(inst.name)} [label="{label}", fillcolor="{fill}", shape={nshape}{pen}];'
        )

    for inst in insts:
        dst = _nid(inst.name)
        for op in inst.operands:
            if op in names:
                out.append(f"  {_nid(op)} -> {dst};")

    out.append("  subgraph cluster_legend {")
    out.append('    label="Op Counts"; style=dashed; fontsize=11;')
    out.append(f'    legend [label="{summary}", shape=note, fillcolor="#FFFDE7", fontsize=8];')
    out.append("  }")
    out.append("}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_dot(dot_str: str, output_path: Path, fmt: str = "png") -> Path:
    dot_file = output_path.with_suffix(".dot")
    dot_file.write_text(dot_str)
    for engine in ("dot", "sfdp"):
        res = subprocess.run(
            [engine, f"-T{fmt}", str(dot_file), "-o", str(output_path)],
            capture_output=True, text=True, timeout=180,
        )
        if res.returncode == 0:
            return output_path
    raise RuntimeError(f"graphviz rendering failed: {res.stderr}")


# ---------------------------------------------------------------------------
# Extraction pipeline
# ---------------------------------------------------------------------------

def extract_hlo_dumps(
    experiment_dir: Path,
    impl: str,
    force_recompile: bool = True,
) -> dict[str, Path | HloModule]:
    """Run experiment and collect pre/post-optimisation HLO.

    Returns ``{"before": HloModule, "after": HloModule}`` plus text files
    saved under ``<experiment>/artifacts/<impl>/``.
    """
    art = experiment_dir / "artifacts" / impl
    art.mkdir(parents=True, exist_ok=True)

    pre_dir = art / "_pre_opt_hlo"
    if pre_dir.exists():
        shutil.rmtree(pre_dir)
    pre_dir.mkdir()

    post_cache = art / "_neuron_cache"
    if post_cache.exists():
        shutil.rmtree(post_cache)
    post_cache.mkdir()

    if force_recompile:
        runner._clean_stale_neffs(experiment_dir)

    run_env = os.environ.copy()
    run_env["NEURON_FRAMEWORK_DEBUG"] = "1"
    run_env["NEURON_CC_FLAGS"] = "--logical-nc-config=1"
    run_env["NEURON_LOGICAL_NC_CONFIG"] = "1"
    run_env["NEURON_COMPILE_CACHE_URL"] = str(post_cache)
    if impl == "nki":
        run_env["XLA_IR_DEBUG"] = "1"
        run_env["XLA_HLO_DEBUG"] = "1"

    wrapper = Path(__file__).resolve().parent / "_hlo_wrapper.py"
    script = experiment_dir / f"run_{impl}.py"
    if not script.exists():
        raise FileNotFoundError(f"Experiment script not found: {script}")

    python = os.environ.get("PYTHON", "python3")
    cmd = [python, str(wrapper), str(script), str(pre_dir)]
    print(f"Running {impl} with HLO capture enabled …")
    proc = subprocess.run(
        cmd, cwd=str(experiment_dir), env=run_env,
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:] if proc.stdout else "", file=sys.stderr)
        print(proc.stderr[-2000:] if proc.stderr else "", file=sys.stderr)
        raise RuntimeError(f"{impl}.py exited with code {proc.returncode}")

    result: dict[str, HloModule] = {}

    # ── Pre-optimisation HLO (text from wrapper) ─────────────────────────
    # Pick the LAST capture: experiments use early mark_steps for input setup
    # and late mark_steps for the actual kernel.
    pre_files = sorted(pre_dir.glob("pre_opt_*.hlo.txt"))
    if pre_files:
        best = pre_files[-1]
        print(f"  Pre-opt HLO: {best.name} ({best.stat().st_size:,} bytes)")
        module = parse_hlo_text(best.read_text())
        if module.entry:
            result["before"] = module
            shutil.copy2(best, art / "hlo_before_optimizations.txt")
    else:
        print("  WARNING: no pre-optimisation HLO captured")

    # ── Post-optimisation HLO (.pb from Neuron compile cache) ────────────
    # Try to match the pre-opt module number (IrToHlo.N ↔ SyncTensorsGraph.N)
    pb_files = list(post_cache.rglob("*.hlo_module.pb"))
    matched_pb: Path | None = None
    if pb_files and "before" in result:
        pre_mod = result["before"].name
        mod_num = re.search(r"\.(\d+)$", pre_mod)
        if mod_num:
            target_suffix = f".{mod_num.group(1)}"
            for pb in pb_files:
                candidate = parse_hlo_pb(pb)
                if candidate.name.endswith(target_suffix):
                    matched_pb = pb
                    break

    if matched_pb is None and pb_files:
        # Fallback: pick the smallest .pb (the simpler kernel, not setup)
        matched_pb = min(pb_files, key=lambda p: p.stat().st_size)

    if matched_pb:
        print(f"  Post-opt HLO: {matched_pb.parent.name}/{matched_pb.name} ({matched_pb.stat().st_size:,} bytes)")
        module = parse_hlo_pb(matched_pb)
        if module.entry:
            result["after"] = module
            (art / "hlo_after_optimizations.txt").write_text(hlo_module_to_text(module))
    else:
        print("  WARNING: no post-optimisation HLO found in compile cache")

    return result


def extract_and_visualize(
    experiment_dir: Path,
    impl: str,
    output_dir: Path | None = None,
    force_recompile: bool = True,
) -> dict[str, Path]:
    """Full pipeline: run → capture HLO → parse → render images."""
    if output_dir is None:
        output_dir = experiment_dir / "artifacts" / impl
    output_dir.mkdir(parents=True, exist_ok=True)

    modules = extract_hlo_dumps(experiment_dir, impl, force_recompile=force_recompile)
    images: dict[str, Path] = {}

    for stage, module in modules.items():
        n = len(module.entry.instructions) if module.entry else 0
        label = "Before" if stage == "before" else "After"
        print(f"\n{label}-optimisation HLO:")
        print(f"  Module : {module.name}")
        if module.entry:
            print(f"  Entry  : {module.entry.name}  ({n} instructions)")
            op_counts: dict[str, int] = {}
            for inst in module.entry.instructions:
                op_counts[inst.opcode] = op_counts.get(inst.opcode, 0) + 1
            top = sorted(op_counts.items(), key=lambda x: -x[1])[:5]
            print(f"  Top ops: {', '.join(f'{o}({c})' for o, c in top)}")

        title = f"{label} Optimisations  [{impl}]"
        dot = hlo_to_dot(module, title=title)
        img = output_dir / f"hlo_{stage}_optimizations.png"
        print(f"  Rendering → {img}")
        render_dot(dot, img)
        images[stage] = img

    if not images:
        print("\nNo HLO images generated. See warnings above.")
    else:
        print(f"\nDone – {len(images)} image(s) saved to {output_dir}")

    return images
