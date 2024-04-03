

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/fr/cfrlofoh52vv5ogosu6ilbn76cdw7nyowvjyawsyyfmc3utvlxgv.py
# Source Nodes: [setitem_85, zeros], Original ATen: [aten.copy, aten.slice_scatter, aten.zeros]
# setitem_85 => copy_85, slice_scatter_23, slice_scatter_24
# zeros => full_8
triton_poi_fused_copy_slice_scatter_zeros_90 = async_compile.triton('triton_poi_fused_copy_slice_scatter_zeros_90', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_scatter_zeros_90', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_slice_scatter_zeros_90(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 41616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 204)
    x0 = xindex % 204
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp9 & tmp5
    tmp11 = tl.load(in_ptr0 + ((-402) + x0 + (200*x1)), tmp10 & xmask, other=0.0)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tl.where(tmp10, tmp12, 0.0)
    tmp14 = 0.0
    tmp15 = tl.where(tmp9, tmp13, tmp14)
    tmp16 = tl.where(tmp5, tmp15, 0.0)
    tmp17 = tl.where(tmp5, tmp16, tmp14)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''')
