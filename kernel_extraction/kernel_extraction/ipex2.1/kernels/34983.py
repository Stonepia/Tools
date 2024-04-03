

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/ai/caiiy5z4oqxefzdxat5u2w2wnpkqe7bdv3b5fc6djabldrhkfawz.py
# Source Nodes: [mul_10, mul_11, mul_12, setitem_2, setitem_3, sub_5, sub_6, truediv_2, truediv_3, zeros_like], Original ATen: [aten.copy, aten.div, aten.mul, aten.slice_scatter, aten.sub, aten.zeros_like]
# mul_10 => mul_10
# mul_11 => mul_11
# mul_12 => mul_12
# setitem_2 => copy_2, slice_scatter_6, slice_scatter_7, slice_scatter_8
# setitem_3 => copy_3, slice_scatter_10, slice_scatter_11, slice_scatter_9
# sub_5 => sub_5
# sub_6 => sub_6
# truediv_2 => div_2
# truediv_3 => div_3
# zeros_like => full
triton_poi_fused_copy_div_mul_slice_scatter_sub_zeros_like_0 = async_compile.triton('triton_poi_fused_copy_div_mul_slice_scatter_sub_zeros_like_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_div_mul_slice_scatter_sub_zeros_like_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_div_mul_slice_scatter_sub_zeros_like_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1082016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5304)
    x3 = xindex
    x1 = (xindex // 26) % 204
    tmp0 = x2
    tmp1 = tl.full([1], 203, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x3), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (15912 + (3*x3)), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (3*x3), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp6 = tmp4 - tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tl.load(in_ptr2 + (x2), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (x1), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 / tmp10
    tmp12 = tl.where(tmp2, tmp11, 0.0)
    tmp13 = 0.0
    tmp14 = tl.where(tmp2, tmp12, tmp13)
    tmp15 = tl.load(in_ptr4 + (15912 + (3*x3)), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr4 + (3*x3), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp17 = tmp15 - tmp16
    tmp18 = tmp3 * tmp17
    tmp19 = tmp18 / tmp10
    tmp20 = tl.where(tmp2, tmp19, 0.0)
    tmp21 = tl.where(tmp2, tmp20, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, xmask)
    tl.store(out_ptr1 + (x3), tmp21, xmask)
''')
