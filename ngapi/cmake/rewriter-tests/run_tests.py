#!/usr/bin/env python3
"""Standalone regression tests for ../RewriteMetalHeaps.py.

Each file under fixtures/ is RAW `slangc -target metal` output (no
post-processing).  The test applies the same EntryPointParams sed step the
build runs (see CompileMetalShaders.cmake) followed by
RewriteMetalHeaps.rewrite(), then asserts structural properties of the
result: no unbounded arrays remain, heap subscripts go through ->data[,
vertex entries regained their EntryPointParams [[buffer(0)]] parameter,
packed_float3 pointer rewrite happened, and so on.

Run from anywhere:
    python3 ngapi/cmake/rewriter-tests/run_tests.py

Regenerating fixtures (from the repository root, slangc on PATH):
    slangc <sample>.slang -target metal -entry <entry> -stage <stage> \
        -I ngapi/include -warnings-disable 39001 -o fixtures/<Name>.metal
Fixture set: Compute(main), Tensor(_add.._tanh), Multithreading(main),
SamplerBench(bench*), TAA(main) as compute; Vertex/TextVertex(main) as
vertex; Pixel/TextPixel(main) as fragment.
"""

import os
import re
import sys

sys.dont_write_bytecode = True  # keep ngapi/cmake/ free of __pycache__

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import RewriteMetalHeaps as rmh  # noqa: E402

ENTRY_SIG_RE = re.compile(r"\[\[(?:kernel|vertex|fragment)\]\][^\n]*")
DIRECT_HEAP_SUBSCRIPT_RE = re.compile(
    r"(?:textureHeap|rwTextureHeap|ngapiSamplerHeap)_\d+\s*\[(?!\[)")

failures = []


def check(name, cond, msg):
    if not cond:
        failures.append(f"{name}: {msg}")


def sed_entry_point_params(src):
    """Python mirror of the sed step in CompileMetalShaders.cmake."""
    return re.sub(r"(EntryPointParams_[0-9]+) constant\*", r"device \1*", src)


def load_and_rewrite(fname):
    with open(os.path.join(HERE, "fixtures", fname), encoding="utf-8") as f:
        return rmh.rewrite(sed_entry_point_params(f.read()))


# ---------------------------------------------------------------------------
# Invariants that must hold for every fixture.
# ---------------------------------------------------------------------------
outputs = {}
fixture_names = sorted(f for f in os.listdir(os.path.join(HERE, "fixtures"))
                       if f.endswith(".metal"))
check("fixtures", len(fixture_names) >= 21, "fixture set went missing")

for fname in fixture_names:
    out = load_and_rewrite(fname)
    outputs[fname] = out

    # No Tier-2 unbounded arrays may survive (wrappers use data[1]).
    check(fname, "[]" not in out, "unbounded array '[]' left in output")

    # Every heap subscript must go through the wrapper's ->data[...].
    check(fname, not DIRECT_HEAP_SUBSCRIPT_RE.search(out),
          "direct heap subscript left in output")

    # constant* EntryPointParams must have been rewritten to device*.
    check(fname, " constant* entryPointParams" not in out,
          "EntryPointParams still in constant address space")

    # Every entry point that uses entry-point uniforms must take the
    # per-draw/per-dispatch param buffer at [[buffer(0)]].
    if "EntryPointParams" in out:
        sigs = ENTRY_SIG_RE.findall(out)
        check(fname, sigs, "no entry-point signature found")
        for sig in sigs:
            check(fname, "EntryPointParams" in sig and "[[buffer(0)]]" in sig,
                  f"entry missing EntryPointParams [[buffer(0)]]: {sig}")

# ---------------------------------------------------------------------------
# Per-fixture key assertions.
# ---------------------------------------------------------------------------
out = outputs["Vertex.metal"]
check("Vertex", "packed_float3 device* vertices_0;" in out,
      "float3 device pointer not rewritten to packed_float3")
check("Vertex", "packed_float3 position" not in out,
      "non-pointer float3 wrongly rewritten")
vtx_sig = next(s for s in ENTRY_SIG_RE.findall(out) if "[[vertex]]" in s)
check("Vertex", re.search(r"device EntryPointParams_0\* \w+ \[\[buffer\(0\)\]\]", vtx_sig),
      f"vertex entry lacks injected EntryPointParams param: {vtx_sig}")
check("Vertex", re.search(r"\(&kernelContext_1\)->entryPointParams_0 = \w+;", out),
      "vertex entry lacks KernelContext wiring for injected param")
check("Vertex", "[[vertex_id]]" in vtx_sig and "[[instance_id]]" in vtx_sig,
      "vertex system-value params not preserved")

out = outputs["TextVertex.metal"]
check("TextVertex", re.search(r"\[\[vertex\]\][^\n]*device EntryPointParams_0\* \w+ \[\[buffer\(0\)\]\]", out),
      "vertex entry lacks injected EntryPointParams param")
check("TextVertex", re.search(r"\(&kernelContext_1\)->entryPointParams_0 = \w+;", out),
      "vertex entry lacks KernelContext wiring for injected param")

for n in ("Pixel.metal", "TextPixel.metal"):
    out = outputs[n]
    frag_sig = next(s for s in ENTRY_SIG_RE.findall(out) if "[[fragment]]" in s)
    check(n, "[[stage_in]]" in frag_sig and "[[position]]" in frag_sig,
          "fragment stage_in/position params not preserved")
    check(n, re.search(r"device ngapi_TexHeap\*\s+textureHeap_1 \[\[buffer\(1\)\]\]", frag_sig),
          f"fragment textureHeap not bound at buffer(1): {frag_sig}")
    check(n, re.search(r"device ngapi_SampHeap\*\s+ngapiSamplerHeap_1 \[\[buffer\(2\)\]\]", frag_sig),
          f"fragment samplerHeap not bound at buffer(2): {frag_sig}")
    check(n, "textureHeap_1->data[" in out,
          "fragment heap parameter subscript not rewritten")
    check(n, "->ngapiSamplerHeap_0->data[" in out,
          "sampler-heap member subscript not rewritten")

out = outputs["TAA.metal"]
check("TAA", "->textureHeap_0->data[" in out,
      "KernelContext textureHeap member subscript not rewritten")
check("TAA", "->rwTextureHeap_0->data[" in out,
      "KernelContext rwTextureHeap member subscript (.write form) not rewritten")
check("TAA", "->ngapiSamplerHeap_0->data[" in out,
      "KernelContext samplerHeap member subscript not rewritten")
for pat in (r"device ngapi_TexHeap\*\s+textureHeap_1 \[\[buffer\(1\)\]\]",
            r"device ngapi_RWTexHeap\*\s+rwTextureHeap_1 \[\[buffer\(2\)\]\]",
            r"device ngapi_SampHeap\*\s+ngapiSamplerHeap_1 \[\[buffer\(3\)\]\]"):
    check("TAA", re.search(pat, out), f"kernel heap binding missing: {pat}")

out = outputs["Compute.metal"]
check("Compute", "->data[" in out, "heap temporaries not rewritten")
check("Compute", "[[thread_position_in_grid]]" in out,
      "thread-position param not preserved")

out = outputs["SamplerBench_benchHardware.metal"]
check("SamplerBench_benchHardware",
      re.search(r"device ngapi_SampHeap\*\s+_S\d+ =", out),
      "local sampler-array temporary not rewritten")
check("SamplerBench_benchHardware",
      re.search(r"_S\d+->data\[int\(0\)\]", out),
      "local sampler temporary subscript not rewritten")

# Tensor entries use no heaps and slangc elides KernelContext entirely:
# the file must pass through with only the sed's device* rewrite applied.
out = outputs["Tensor_add.metal"]
check("Tensor_add", "KernelContext" not in out and "ngapi_TexHeap" not in out,
      "heap machinery wrongly injected into a heap-free shader")
check("Tensor_add", "device EntryPointParams_0* entryPointParams_0 [[buffer(0)]]" in out,
      "kernel EntryPointParams binding changed")

# ---------------------------------------------------------------------------
# Synthetic: word-boundary safety of the heap-variable subscript rewrite
# (_S1 must not match _S10 or arr_S1).
# ---------------------------------------------------------------------------
synthetic = """
struct EntryPointParams_0 { int x_0; };
struct KernelContext_0
{
    texture2d<float, access::sample>  textureHeap_0[];
    EntryPointParams_0 constant* entryPointParams_0;
};
[[kernel]] void main_0(uint3 t_0 [[thread_position_in_grid]], texture2d<float, access::sample>  textureHeap_1[], EntryPointParams_0 constant* entryPointParams_1 [[buffer(0)]])
{
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->textureHeap_0 = textureHeap_1;
    (&kernelContext_1)->entryPointParams_0 = entryPointParams_1;
    texture2d<float, access::sample>  _S1[] = (&kernelContext_1)->textureHeap_0;
    int _S10[4];
    int arr_S1[4];
    int a_0 = _S10[0] + arr_S1[1];
    float4 c_0 = _S1[0].read(uint2(0u, 0u));
}
"""
out = rmh.rewrite(sed_entry_point_params(synthetic))
check("synthetic", "_S1->data[0]" in out, "heap temporary _S1 subscript not rewritten")
check("synthetic", "_S10[0]" in out, "_S10 corrupted by _S1 prefix collision")
check("synthetic", "arr_S1[1]" in out, "arr_S1 corrupted by _S1 suffix collision")
check("synthetic", "_S10->data" not in out and "arr_S1->data" not in out,
      "non-heap arrays wrongly rewritten")

# Synthetic: packed-vector rewrite must hit pointers only, in both spellings.
packed_src = """
struct V_0
{
    float3 normal_0;
    float3 device* verts_0;
    device const uint3* idx_0;
    int3 thread* tmp_0;
    float3x3 device* mats_0;
};
"""
out = rmh.rewrite(packed_src)
check("packed", "float3 normal_0;" in out, "non-pointer field wrongly packed")
check("packed", "packed_float3 device* verts_0;" in out, "postfix device pointer not packed")
check("packed", "device const packed_uint3* idx_0;" in out, "prefix device pointer not packed")
check("packed", "int3 thread* tmp_0;" in out, "thread pointer wrongly packed")
check("packed", "float3x3 device* mats_0;" in out, "matrix pointer wrongly packed")

# ---------------------------------------------------------------------------
if failures:
    print(f"FAIL ({len(failures)} failures across {len(fixture_names)} fixtures)")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
print(f"PASS ({len(fixture_names)} fixtures + synthetic checks)")
