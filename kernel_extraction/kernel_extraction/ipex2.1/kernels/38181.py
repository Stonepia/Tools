

# Original file: ./basic_gnn_sage___60.0/basic_gnn_sage___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/le/cle5w5bmy66ggbqza4mbm4xuvlg5snscqc2aufg6bwyhfetbwbiq.py
# Source Nodes: [new_ones, new_zeros, scatter_add_], Original ATen: [aten.new_ones, aten.new_zeros, aten.scatter_add]
# new_ones => full_default_1
# new_zeros => full_default
# scatter_add_ => scatter_add
triton_poi_fused_new_ones_new_zeros_scatter_add_3 = async_compile.triton('triton_poi_fused_new_ones_new_zeros_scatter_add_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_ones_new_zeros_scatter_add_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_new_ones_new_zeros_scatter_add_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (200000 + x0), xmask)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 10000, tmp0)
    # tl.device_assert(((0 <= tmp1) & (tmp1 < 10000)) | ~xmask, "index out of bounds: 0 <= tmp1 < 10000")
    tmp2 = 1.0
    tl.atomic_add(out_ptr0 + (tl.broadcast_to(tmp1, [XBLOCK])), tmp2, xmask)
''')
