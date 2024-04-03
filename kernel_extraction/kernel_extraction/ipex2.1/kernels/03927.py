

# Original file: ./basic_gnn_edgecnn___60.0/basic_gnn_edgecnn___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/7b/c7bv7dgk6q5e4rmhsahwzifa2p2c2pf7ufkbbj3jamly7uqbne4m.py
# Source Nodes: [index_select, index_select_1, sub], Original ATen: [aten.index_select, aten.sub]
# index_select => index
# index_select_1 => index_1
# sub => sub
triton_poi_fused_index_select_sub_0 = async_compile.triton('triton_poi_fused_index_select_sub_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_sub_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_index_select_sub_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12800000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 64)
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (200000 + x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.where(tmp0 < 0, tmp0 + 10000, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 10000), "index out of bounds: 0 <= tmp1 < 10000")
    tmp2 = tl.load(in_ptr1 + (x0 + (64*tmp1)), None)
    tmp4 = tl.where(tmp3 < 0, tmp3 + 10000, tmp3)
    # tl.device_assert((0 <= tmp4) & (tmp4 < 10000), "index out of bounds: 0 <= tmp4 < 10000")
    tmp5 = tl.load(in_ptr1 + (x0 + (64*tmp4)), None)
    tmp6 = tmp5 - tmp2
    tl.store(out_ptr0 + (x0 + (128*x1)), tmp2, None)
    tl.store(out_ptr1 + (x0 + (128*x1)), tmp6, None)
''')
