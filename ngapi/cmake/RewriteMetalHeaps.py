#!/usr/bin/env python3
"""
Post-process Slang-generated MSL so it compiles with Metal's runtime
compiler (newLibraryWithSource:).

Slang emits Tier-2 argument buffer syntax:
  • Multiple flexible array members in KernelContext_N struct
  • Unbounded resource arrays (textureHeap_N[], sampler ngapiSamplerHeap_N[])
    as kernel parameters without explicit [[buffer(N)]] annotations

Both forms are rejected by the runtime compiler.  We rewrite them to valid
single-flexible-array wrapper structs with explicit buffer bindings:

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
      device EntryPointParams_N* ep [[buffer(0)]],
      device ngapi_TexHeap*    textureHeap_N   [[buffer(1)]],
      device ngapi_RWTexHeap*  rwTextureHeap_N [[buffer(2)]],
      device ngapi_SampHeap*   ngapiSamplerHeap_N [[buffer(3)]])

Local texture-array temporaries and sampler-heap subscript accesses are
updated to match (e.g. _S8[idx] → _S8->data[idx]).

Files that do not contain a KernelContext struct are passed through unchanged.
"""

import re
import sys

WRAPPERS = (
    "struct ngapi_TexHeap   { texture2d<float, access::sample>       data[1]; };\n"
    "struct ngapi_RWTexHeap { texture2d<float, access::read_write>   data[1]; };\n"
    "struct ngapi_SampHeap  { sampler                                data[1]; };\n"
    "\n"
)


def _rewrite_kernelctx_body(m):
    s = m.group(0)
    s = re.sub(r"texture2d<float,\s*access::sample>\s+(\w+)\[\];",
               r"device ngapi_TexHeap*    \1;", s)
    s = re.sub(r"texture2d<float,\s*access::read_write>\s+(\w+)\[\];",
               r"device ngapi_RWTexHeap*  \1;", s)
    s = re.sub(r"\bsampler\s+(\w+)\[\];",
               r"device ngapi_SampHeap*   \1;", s)
    return s


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


def _rewrite_kernel_sig(m):
    text = m.group(0)  # "[[kernel]] void NAME(...)\n{"
    sig_line = text.split("\n", 1)[0]

    fname_m = re.search(r"\[\[kernel\]\]\s+void\s+(\w+)\s*\(", sig_line)
    fname = fname_m.group(1) if fname_m else "main_0"

    p_start = sig_line.index("(") + 1
    p_end = sig_line.rindex(")")
    params = _split_params(sig_line[p_start:p_end])

    tp_param = None
    ep_param = None
    tex_name = None
    rwtex_name = None
    samp_name = None

    for p in params:
        if "[[thread_position_in_grid]]" in p:
            tp_param = p
        elif "EntryPointParams" in p and "[[buffer" in p:
            ep_param = p
        elif "texture2d<float, access::sample>" in p or "texture2d<float,access::sample>" in p:
            nm = re.search(r"(\w+)\[\]", p)
            if nm:
                tex_name = nm.group(1)
        elif "texture2d<float, access::read_write>" in p or "texture2d<float,access::read_write>" in p:
            nm = re.search(r"(\w+)\[\]", p)
            if nm:
                rwtex_name = nm.group(1)
        elif re.match(r"sampler\s", p.strip()) and "[]" in p:
            nm = re.search(r"(\w+)\[\]", p)
            if nm:
                samp_name = nm.group(1)

    new_params = []
    if tp_param:
        new_params.append(tp_param)
    if ep_param:
        new_params.append(ep_param)
    buf_idx = 1
    if tex_name:
        new_params.append(f"device ngapi_TexHeap*    {tex_name} [[buffer({buf_idx})]]")
        buf_idx += 1
    if rwtex_name:
        new_params.append(f"device ngapi_RWTexHeap*  {rwtex_name} [[buffer({buf_idx})]]")
        buf_idx += 1
    if samp_name:
        new_params.append(f"device ngapi_SampHeap*   {samp_name} [[buffer({buf_idx})]]")
        buf_idx += 1

    return f"[[kernel]] void {fname}({', '.join(new_params)})\n{{"


def rewrite(src):
    # Skip files that don't use NGAPI resource heaps.
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

    # 3. Rewrite kernel signature: unbounded array params → explicit buffer bindings.
    src = re.sub(
        r"\[\[kernel\]\][^\n]+\n\{",
        _rewrite_kernel_sig,
        src,
    )

    # 4. Rewrite local texture/rw-texture array declarations; track variable names.
    tex_vars: set = set()
    rwtex_vars: set = set()

    def _cap_tex(m):
        tex_vars.add(m.group(1))
        return f"device ngapi_TexHeap*    {m.group(1)} ="

    def _cap_rwtex(m):
        rwtex_vars.add(m.group(1))
        return f"device ngapi_RWTexHeap*  {m.group(1)} ="

    src = re.sub(r"texture2d<float,\s*access::sample>\s+(\w+)\[\]\s*=", _cap_tex, src)
    src = re.sub(r"texture2d<float,\s*access::read_write>\s+(\w+)\[\]\s*=", _cap_rwtex, src)

    # 5. Replace VARNAME[...] → VARNAME->data[...] for tracked heap variables.
    for var in tex_vars:
        src = src.replace(f"{var}[", f"{var}->data[")
    for var in rwtex_vars:
        src = src.replace(f"{var}[", f"{var}->data[")

    # 6. Rewrite sampler-heap subscript accesses via a KernelContext pointer.
    #    ->ngapiSamplerHeap_N[idx]  →  ->ngapiSamplerHeap_N->data[idx]
    src = re.sub(r"->(\bngapiSamplerHeap_\d+)\[", r"->\1->data[", src)

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
