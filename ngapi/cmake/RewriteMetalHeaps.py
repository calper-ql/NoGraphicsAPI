#!/usr/bin/env python3
"""
Post-process Slang-generated MSL so it compiles with Metal's runtime
compiler (newLibraryWithSource:) and matches the NGAPI Metal ABI.

Slang emits Tier-2 argument buffer syntax (and a few stage-specific bugs):
  • Multiple flexible array members in KernelContext_N struct
  • Unbounded resource arrays (textureHeap_N[], sampler ngapiSamplerHeap_N[])
    as entry-point parameters without explicit [[buffer(N)]] annotations —
    on [[kernel]], [[vertex]] and [[fragment]] entries alike
  • -stage vertex DROPS the EntryPointParams uniform parameter entirely
    (the local KernelContext is read uninitialized)
  • `float3*` pointers come out as `float3 device*` (16-byte stride) while
    the CPU side shares tightly packed 12-byte float3 data

These are rejected (or miscompile) by the runtime compiler.  We rewrite
them to valid single-flexible-array wrapper structs with explicit buffer
bindings:

  struct ngapi_TexHeap   { texture2d<float, access::sample>     data[]; };
  struct ngapi_RWTexHeap { texture2d<float, access::read_write> data[]; };
  struct ngapi_SampHeap  { sampler                              data[]; };

  struct KernelContext_N {
      device ngapi_TexHeap*    textureHeap_N;
      device ngapi_RWTexHeap*  rwTextureHeap_N;
      device ngapi_SampHeap*   ngapiSamplerHeap_N;
      device EntryPointParams_N* entryPointParams_N;
  };

  [[kernel]] void main_N(
      uint3 threadId [[thread_position_in_grid]],
      device ngapi_TexHeap*    textureHeap_N   [[buffer(1)]],
      device ngapi_SampHeap*   ngapiSamplerHeap_N [[buffer(3)]],
      device ngapi_RWTexHeap*  rwTextureHeap_N [[buffer(2)]],
      device EntryPointParams_N* ep [[buffer(0)]])

Buffer index convention (identical for every stage; vertex and fragment
buffer indices are independent namespaces on Metal):
  entryPointParams = buffer(0), then — among the heaps that are present —
  textureHeap = 1, rwTextureHeap = next, samplerHeap = next.
Non-heap parameters ([[stage_in]], [[position]], [[vertex_id]],
[[instance_id]], [[thread_position_in_grid]], ...) are kept verbatim.

Heap subscripts in function bodies are updated to match, word-boundary
safe (_S1 never matches _S10):
  _S8[idx]                      → _S8->data[idx]
  ctx->textureHeap_N[idx]       → ctx->textureHeap_N->data[idx]
  ctx->rwTextureHeap_N[idx]     → ctx->rwTextureHeap_N->data[idx]   (.write/.read too)
  ctx->ngapiSamplerHeap_N[idx]  → ctx->ngapiSamplerHeap_N->data[idx]

[[vertex]] entries that lost parameters get them re-injected: any
KernelContext member that the body declares locally but never wires from
a parameter gains a `device T* name [[buffer(i)]]` parameter plus a
`(&kernelContext)->member = name;` assignment, mirroring what slangc
already emits for fragment/kernel stages.

Pointer DECLARATIONS of 3-component vectors (float3/uint3/int3, in either
`float3 device*` or `device float3*` spelling, optionally const) are
rewritten to packed_float3/packed_uint3/packed_int3 so pointer arithmetic
matches the packed CPU layout; packed loads implicitly convert to the
aligned vector types.  Non-pointer struct fields and float3x3-style
matrix types are never touched.

Files that do not contain a KernelContext struct only get the
packed-vector rewrite.
"""

import re
import sys

WRAPPERS = (
    "struct ngapi_TexHeap   { texture2d<float, access::sample>       data[1]; };\n"
    "struct ngapi_RWTexHeap { texture2d<float, access::read_write>   data[1]; };\n"
    "struct ngapi_SampHeap  { sampler                                data[1]; };\n"
    "\n"
)

# Heap kinds in buffer-index allocation order (entryPointParams is always 0).
HEAP_ORDER = ("tex", "rwtex", "samp")

_HEAP_PARAM_DECL = {
    "tex":   "device ngapi_TexHeap*    {name} [[buffer({idx})]]",
    "rwtex": "device ngapi_RWTexHeap*  {name} [[buffer({idx})]]",
    "samp":  "device ngapi_SampHeap*   {name} [[buffer({idx})]]",
}

_PACKED_VEC = {"float3": "packed_float3", "uint3": "packed_uint3", "int3": "packed_int3"}


def _rewrite_packed_vec_pointers(src):
    """Rewrite 3-component vector pointer declarations to packed_* pointees.

    Handles both slangc's postfix address-space form ('float3 device*') and
    the conventional prefix form ('device float3*'), with optional const.
    Only pointer declarations match — plain struct fields have no 'device'
    + '*', and matrix types like float3x3 fail the required whitespace
    after the vector token.
    """
    src = re.sub(
        r"\b(float3|uint3|int3)(\s+(?:const\s+)?device\s*\*)",
        lambda m: _PACKED_VEC[m.group(1)] + m.group(2),
        src,
    )
    src = re.sub(
        r"\b(device\s+(?:const\s+)?)(float3|uint3|int3)(\s*\*)",
        lambda m: m.group(1) + _PACKED_VEC[m.group(2)] + m.group(3),
        src,
    )
    return src


def _rewrite_kernelctx_body(m):
    s = m.group(0)
    s = re.sub(r"texture2d<float,\s*access::sample>\s+(\w+)\[\];",
               r"device ngapi_TexHeap*    \1;", s)
    s = re.sub(r"texture2d<float,\s*access::read_write>\s+(\w+)\[\];",
               r"device ngapi_RWTexHeap*  \1;", s)
    s = re.sub(r"\bsampler\s+(\w+)\[\];",
               r"device ngapi_SampHeap*   \1;", s)
    return s


def _parse_kernel_contexts(src):
    """Map KernelContext number → member dict, parsed AFTER the struct rewrite.

    Members: 'ep' → (EntryPointParams struct name, member name),
             'tex'/'rwtex'/'samp' → member name.
    """
    ctxs = {}
    for m in re.finditer(r"struct KernelContext_(\d+)\s*\{(.*?)\};", src, re.DOTALL):
        body = m.group(2)
        members = {}
        em = re.search(r"device\s+(EntryPointParams_\d+)\*\s+(\w+)\s*;", body)
        if em:
            members["ep"] = (em.group(1), em.group(2))
        for kind, pat in (("tex",   r"device\s+ngapi_TexHeap\*\s+(\w+)\s*;"),
                          ("rwtex", r"device\s+ngapi_RWTexHeap\*\s+(\w+)\s*;"),
                          ("samp",  r"device\s+ngapi_SampHeap\*\s+(\w+)\s*;")):
            km = re.search(pat, body)
            if km:
                members[kind] = km.group(1)
        ctxs[m.group(1)] = members
    return ctxs


def _split_params(params_str):
    """Split a parameter list string on top-level commas.

    Treats '([<' as openers and ')]>' as closers so commas inside
    template arguments like texture2d<float, access::sample> are ignored.
    """
    params = []
    depth = 0
    cur = []
    for ch in params_str:
        if ch in "([<":
            depth += 1
            cur.append(ch)
        elif ch in ")]>":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            params.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        params.append("".join(cur).strip())
    return params


def _classify_heap_param(p):
    """Return ('tex'|'rwtex'|'samp', name) for unbounded heap-array
    parameters, else None."""
    nm = re.search(r"(\w+)\s*\[\s*\]", p)
    if not nm:
        return None
    if re.search(r"texture2d<float,\s*access::sample>", p):
        return ("tex", nm.group(1))
    if re.search(r"texture2d<float,\s*access::read_write>", p):
        return ("rwtex", nm.group(1))
    if re.match(r"sampler\s", p.strip()):
        return ("samp", nm.group(1))
    return None


def _fresh_name(member, src):
    """Pick a slangc-style name for an injected parameter wiring `member`
    (entryPointParams_0 → entryPointParams_1), avoiding collisions."""
    m = re.match(r"(.*_)(\d+)$", member)
    base, n = (m.group(1), int(m.group(2))) if m else (member + "_", 0)
    cand = f"{base}{n + 1}"
    while re.search(rf"\b{re.escape(cand)}\b", src):
        n += 1
        cand = f"{base}{n + 1}"
    return cand


def _find_matching_brace(src, open_idx):
    depth = 0
    for i in range(open_idx, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("RewriteMetalHeaps: unbalanced braces in entry-point body")


def _rewrite_one_entry(sig, body, ctxs, heap_vars, full_src):
    """Rewrite one entry-point signature (+ body wiring when parameters
    were dropped by slangc).  Returns (new_sig, new_body)."""
    if "(" not in sig:
        return sig, body
    p_start = sig.index("(") + 1
    p_end = sig.rindex(")")
    head = sig[:p_start - 1]

    entries = []        # ("heap", kind, name) | ("keep", text), original order
    sig_kinds = set()
    has_ep_param = False
    for p in _split_params(sig[p_start:p_end]):
        c = _classify_heap_param(p)
        if c:
            entries.append(("heap",) + c)
            sig_kinds.add(c[0])
            heap_vars.add(c[1])
        else:
            if "EntryPointParams" in p and "[[buffer" in p:
                has_ep_param = True
            entries.append(("keep", p))

    # Re-inject parameters slangc dropped (-stage vertex): any KernelContext
    # member whose local context is declared but never wired from a parameter.
    injected = []       # (kind, param name)
    wiring = []         # (ctx var, member, param name)
    ctx_decl = re.search(r"thread\s+KernelContext_(\d+)\s+(\w+)\s*;", body)
    ep_struct = None
    if ctx_decl:
        members = ctxs.get(ctx_decl.group(1), {})
        ctx_var = ctx_decl.group(2)
        for kind in ("ep",) + HEAP_ORDER:
            if kind not in members:
                continue
            if kind == "ep":
                ep_struct, member = members["ep"]
                if has_ep_param:
                    continue
            else:
                member = members[kind]
                if kind in sig_kinds:
                    continue
            if re.search(rf"->\s*{re.escape(member)}\s*=", body):
                continue        # already wired by slangc
            pname = _fresh_name(member, full_src)
            injected.append((kind, pname))
            wiring.append((ctx_var, member, pname))
            if kind != "ep":
                sig_kinds.add(kind)
                heap_vars.add(pname)

    # Allocate heap buffer indices: tex=1, rwtex=next, samp=next among present.
    kind_index = {}
    idx = 1
    for kind in HEAP_ORDER:
        if kind in sig_kinds:
            kind_index[kind] = idx
            idx += 1

    new_params = []
    for e in entries:
        if e[0] == "keep":
            new_params.append(e[1])
        else:
            _, kind, name = e
            new_params.append(_HEAP_PARAM_DECL[kind].format(name=name, idx=kind_index[kind]))
    for kind, pname in injected:
        if kind == "ep":
            new_params.append(f"device {ep_struct}* {pname} [[buffer(0)]]")
        else:
            new_params.append(_HEAP_PARAM_DECL[kind].format(name=pname, idx=kind_index[kind]))

    new_sig = f"{head}({', '.join(new_params)})"

    new_body = body
    if ctx_decl and wiring:
        insert_at = ctx_decl.end()
        lines = "".join(f"\n\n    (&{var})->{member} = {name};"
                        for var, member, name in wiring)
        new_body = body[:insert_at] + lines + body[insert_at:]
    return new_sig, new_body


_ENTRY_RE = re.compile(r"\[\[(?:kernel|vertex|fragment)\]\][^\n]*\n\{")


def _rewrite_entry_points(src, ctxs, heap_vars):
    out = []
    last = 0
    for m in _ENTRY_RE.finditer(src):
        close = _find_matching_brace(src, m.end() - 1)
        sig = src[m.start():m.end() - 2]      # signature line, sans "\n{"
        body = src[m.end():close]
        new_sig, new_body = _rewrite_one_entry(sig, body, ctxs, heap_vars, src)
        out.append(src[last:m.start()])
        out.append(new_sig)
        out.append("\n{")
        out.append(new_body)
        last = close
    out.append(src[last:])
    return "".join(out)


def rewrite(src):
    # 0. Packed 3-component vector pointers (Metal-only CPU/GPU ABI fix);
    #    applies even to files that don't use NGAPI resource heaps.
    src = _rewrite_packed_vec_pointers(src)

    # Skip files that don't use NGAPI resource heaps / entry-point params.
    if "KernelContext_" not in src:
        return src

    # 1. Insert wrapper structs before the first KernelContext struct definition.
    src = re.sub(r"(\bstruct KernelContext_\d+\b)", WRAPPERS + r"\1", src, count=1)

    # 2. Rewrite KernelContext struct: flexible-array members → device pointers.
    src = re.sub(
        r"(struct KernelContext_\d+\s*\{[^}]*\})",
        _rewrite_kernelctx_body,
        src,
        flags=re.DOTALL,
    )

    ctxs = _parse_kernel_contexts(src)

    # 3. Rewrite entry-point signatures ([[kernel]]/[[vertex]]/[[fragment]]):
    #    unbounded heap arrays → explicit wrapper-struct buffer bindings, and
    #    re-inject parameters slangc dropped (vertex EntryPointParams).
    heap_vars: set = set()
    src = _rewrite_entry_points(src, ctxs, heap_vars)

    # 4. Rewrite local texture/rw-texture/sampler array declarations;
    #    track variable names.
    def _cap_tex(m):
        heap_vars.add(m.group(1))
        return f"device ngapi_TexHeap*    {m.group(1)} ="

    def _cap_rwtex(m):
        heap_vars.add(m.group(1))
        return f"device ngapi_RWTexHeap*  {m.group(1)} ="

    def _cap_samp(m):
        heap_vars.add(m.group(1))
        return f"device ngapi_SampHeap*   {m.group(1)} ="

    src = re.sub(r"texture2d<float,\s*access::sample>\s+(\w+)\[\]\s*=", _cap_tex, src)
    src = re.sub(r"texture2d<float,\s*access::read_write>\s+(\w+)\[\]\s*=", _cap_rwtex, src)
    src = re.sub(r"\bsampler\s+(\w+)\[\]\s*=", _cap_samp, src)

    # 5. Replace VARNAME[...] → VARNAME->data[...] for tracked heap variables
    #    (word-boundary safe: _S1 never matches _S10; the (?!\[) lookahead
    #    keeps [[buffer(N)]] attributes in rewritten signatures intact).
    for var in sorted(heap_vars):
        src = re.sub(rf"\b{re.escape(var)}\s*\[(?!\[)", f"{var}->data[", src)

    # 6. Rewrite heap-member subscripts through a KernelContext pointer/value:
    #    ->textureHeap_N[idx]      → ->textureHeap_N->data[idx]
    #    ->rwTextureHeap_N[idx]    → ->rwTextureHeap_N->data[idx]
    #    ->ngapiSamplerHeap_N[idx] → ->ngapiSamplerHeap_N->data[idx]
    src = re.sub(
        r"(->|\.)((?:textureHeap|rwTextureHeap|ngapiSamplerHeap)_\d+)\s*\[(?!\[)",
        r"\1\2->data[",
        src,
    )

    return src


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.metal> [output.metal]", file=sys.stderr)
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else inp
    with open(inp, encoding="utf-8") as f:
        src = f.read()
    result = rewrite(src)
    with open(out, "w", encoding="utf-8") as f:
        f.write(result)
