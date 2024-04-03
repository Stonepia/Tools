

# Original file: ./basic_gnn_sage___60.0/basic_gnn_sage___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/os/cosmdytniogiemycsgtucknh7wxse2ku57qhimyzmw4zcylj2bxf.py
# Source Nodes: [index_select_1, l__mod___act, new_zeros_1, scatter_add__3], Original ATen: [aten.index_select, aten.new_zeros, aten.relu, aten.scatter_add]
# index_select_1 => index_1
# l__mod___act => relu
# new_zeros_1 => full_default_2
# scatter_add__3 => scatter_add_3
triton_poi_fused_index_select_new_zeros_relu_scatter_add_5 = async_compile.triton('triton_poi_fused_index_select_new_zeros_relu_scatter_add_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_new_zeros_relu_scatter_add_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_index_select_new_zeros_relu_scatter_add_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12800000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 64)
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (200000 + x1), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.where(tmp0 < 0, tmp0 + 10000, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 10000), "index out of bounds: 0 <= tmp1 < 10000")
    tmp3 = tl.where(tmp2 < 0, tmp2 + 10000, tmp2)
    # tl.device_assert((0 <= tmp3) & (tmp3 < 10000), "index out of bounds: 0 <= tmp3 < 10000")
    tmp4 = tl.load(in_ptr1 + (x0 + (64*tmp3)), None)
    tmp5 = triton_helpers.maximum(0, tmp4)
    tl.atomic_add(out_ptr0 + (x0 + (64*tmp1)), tmp5, None)
''')
