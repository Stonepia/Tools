

# Original file: ./vision_maskrcnn__53_inference_93.33/vision_maskrcnn__53_inference_93.33.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/gn/cgnz6jlb4yob3gcz6buvbog7o7hxijrxklkcqi6cg6wcz4ybub3f.py
# Source Nodes: [add, add_1, add_2, add_3, imul, imul_1, mul, mul_1, mul_2, mul_3, setitem, setitem_1, setitem_2, setitem_3, sub, sub_1, sub_2, sub_3, zeros_like], Original ATen: [aten.add, aten.copy, aten.mul, aten.select_scatter, aten.slice, aten.slice_scatter, aten.sub, aten.zeros_like]
# add => add
# add_1 => add_1
# add_2 => add_2
# add_3 => add_3
# imul => mul_4
# imul_1 => mul_5
# mul => mul
# mul_1 => mul_1
# mul_2 => mul_2
# mul_3 => mul_3
# setitem => copy, select_scatter, slice_9, slice_scatter
# setitem_1 => copy_1, select_scatter_1, slice_13, slice_scatter_1
# setitem_2 => copy_2, select_scatter_2, slice_17, slice_scatter_2
# setitem_3 => copy_3, select_scatter_3, slice_21
# sub => sub
# sub_1 => sub_1
# sub_2 => sub_2
# sub_3 => sub_3
# zeros_like => full
triton_poi_fused_add_copy_mul_select_scatter_slice_slice_scatter_sub_zeros_like_1 = async_compile.triton('triton_poi_fused_add_copy_mul_select_scatter_slice_slice_scatter_sub_zeros_like_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[256], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy_mul_select_scatter_slice_slice_scatter_sub_zeros_like_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_copy_mul_select_scatter_slice_slice_scatter_sub_zeros_like_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = (xindex // 4)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (3 + (4*x1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1 + (4*x1)), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (2 + (4*x1)), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = 0.5
    tmp7 = tmp5 * tmp6
    tmp8 = tmp3 - tmp4
    tmp9 = tmp8 * tmp6
    tmp10 = 1.0714285714285714
    tmp11 = tmp9 * tmp10
    tmp12 = tmp7 + tmp11
    tmp13 = tl.full([1], 1, tl.int32)
    tmp14 = tmp0 == tmp13
    tmp15 = tmp7 - tmp11
    tmp16 = tl.full([1], 2, tl.int32)
    tmp17 = tmp0 == tmp16
    tmp20 = tmp18 + tmp19
    tmp21 = tmp20 * tmp6
    tmp22 = tmp18 - tmp19
    tmp23 = tmp22 * tmp6
    tmp24 = tmp23 * tmp10
    tmp25 = tmp21 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = tmp0 == tmp26
    tmp28 = tmp21 - tmp24
    tmp29 = 0.0
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tl.where(tmp17, tmp25, tmp30)
    tmp32 = tl.where(tmp14, tmp15, tmp31)
    tmp33 = tl.where(tmp2, tmp12, tmp32)
    tl.store(out_ptr0 + (x2), tmp33, xmask)
''')
