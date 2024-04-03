

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/tu/ctuw7zvst64thct3p65mqw5ormav5ze6e5wp6xcblhcbowx2cb4s.py
# Source Nodes: [add_67, mul_182, setitem_37, setitem_38, truediv_40], Original ATen: [aten.add, aten.copy, aten.div, aten.mul, aten.slice_scatter]
# add_67 => add_83
# mul_182 => mul_182
# setitem_37 => copy_37, slice_scatter_186, slice_scatter_187, slice_scatter_188
# setitem_38 => slice_scatter_190
# truediv_40 => div_40
triton_poi_fused_add_copy_div_mul_slice_scatter_52 = async_compile.triton('triton_poi_fused_add_copy_div_mul_slice_scatter_52', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_ptr4', 'out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy_div_mul_slice_scatter_52', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_copy_div_mul_slice_scatter_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1082016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5304)
    x3 = xindex
    x1 = (xindex // 26) % 204
    x0 = xindex % 26
    tmp31 = tl.load(in_ptr4 + (x3), xmask).to(tl.float32)
    tmp0 = x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-10608) + x3), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = x1
    tmp9 = tmp8 >= tmp1
    tmp10 = tmp8 < tmp3
    tmp11 = tmp9 & tmp10
    tmp12 = tmp11 & tmp5
    tmp13 = x0
    tmp14 = tl.full([1], 25, tl.int64)
    tmp15 = tmp13 < tmp14
    tmp16 = tmp15 & tmp12
    tmp17 = tl.load(in_ptr1 + (x3), tmp16 & xmask, other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr2 + (x2), tmp16 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = 4.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp17 / tmp20
    tmp22 = tl.load(in_ptr3 + ((-10050) + x0 + (25*x1) + (5000*x2)), tmp16 & xmask, other=0.0).to(tl.float32)
    tmp23 = tmp21 + tmp22
    tmp24 = tl.where(tmp16, tmp23, 0.0)
    tmp25 = tl.load(in_ptr4 + (x3), tmp12 & xmask, other=0.0).to(tl.float32)
    tmp26 = tl.where(tmp15, tmp24, tmp25)
    tmp27 = tl.where(tmp12, tmp26, 0.0)
    tmp28 = tl.load(in_ptr4 + (x3), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp29 = tl.where(tmp11, tmp27, tmp28)
    tmp30 = tl.where(tmp5, tmp29, 0.0)
    tmp32 = tl.where(tmp5, tmp30, tmp31)
    tmp33 = tl.where(tmp5, tmp7, tmp32)
    tl.store(out_ptr0 + (x3), tmp33, xmask)
''')
