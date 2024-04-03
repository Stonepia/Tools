

# Original file: ./basic_gnn_gcn__23_inference_63.3/basic_gnn_gcn__23_inference_63.3_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/df/cdfy3mer4i3qj5h6srl74rfutzoebjbv45xahncovyibkfskesx2.py
# Source Nodes: [index_select, mul, new_zeros, scatter_add_], Original ATen: [aten.index_select, aten.mul, aten.new_zeros, aten.scatter_add]
# index_select => index
# mul => mul
# new_zeros => full_default
# scatter_add_ => scatter_add
triton_poi_fused_index_select_mul_new_zeros_scatter_add_2 = async_compile.triton('triton_poi_fused_index_select_mul_new_zeros_scatter_add_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp16', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_mul_new_zeros_scatter_add_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_index_select_mul_new_zeros_scatter_add_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13439552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 64)
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (209993 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.where(tmp0 < 0, tmp0 + 10000, tmp0)
    # tl.device_assert(((0 <= tmp1) & (tmp1 < 10000)) | ~xmask, "index out of bounds: 0 <= tmp1 < 10000")
    tmp4 = tl.where(tmp3 < 0, tmp3 + 10000, tmp3)
    # tl.device_assert(((0 <= tmp4) & (tmp4 < 10000)) | ~xmask, "index out of bounds: 0 <= tmp4 < 10000")
    tmp5 = tl.load(in_ptr2 + (x0 + (64*tmp4)), xmask).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp2 * tmp6
    tl.atomic_add(out_ptr0 + (x0 + (64*tmp1)), tmp7, xmask)
''')
