

# Original file: ./basic_gnn_edgecnn___60.0/basic_gnn_edgecnn___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/hc/chcn22efi6h5l5uxuu5zszdr2wghj74pkovieoe7ciabzx2jpw2n.py
# Source Nodes: [index_select_2, index_select_3, l__mod___convs_0_nn_act_1, sub_1], Original ATen: [aten.index_select, aten.relu, aten.sub]
# index_select_2 => index_2
# index_select_3 => index_3
# l__mod___convs_0_nn_act_1 => relu_1
# sub_1 => sub_1
triton_poi_fused_index_select_relu_sub_3 = async_compile.triton('triton_poi_fused_index_select_relu_sub_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_relu_sub_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_index_select_relu_sub_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12800000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 64)
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (200000 + x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.where(tmp0 < 0, tmp0 + 10000, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 10000), "index out of bounds: 0 <= tmp1 < 10000")
    tmp2 = tl.load(in_ptr1 + (x0 + (64*tmp1)), None)
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp5 = tl.where(tmp4 < 0, tmp4 + 10000, tmp4)
    # tl.device_assert((0 <= tmp5) & (tmp5 < 10000), "index out of bounds: 0 <= tmp5 < 10000")
    tmp6 = tl.load(in_ptr1 + (x0 + (64*tmp5)), None)
    tmp7 = triton_helpers.maximum(0, tmp6)
    tmp8 = tmp7 - tmp3
    tl.store(out_ptr0 + (x0 + (128*x1)), tmp3, None)
    tl.store(out_ptr1 + (x0 + (128*x1)), tmp8, None)
''')
