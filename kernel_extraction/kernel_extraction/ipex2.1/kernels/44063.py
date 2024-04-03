

# Original file: ./yolov3___60.0/yolov3___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/kj/ckjcoubkx5iyvk5mtpbxybhya6iw4ypdjjgjvnsrdprgfdquohm3.py
# Source Nodes: [add_23, clone, contiguous, exp, imul, mul, setitem, setitem_1, sigmoid], Original ATen: [aten.add, aten.clone, aten.copy, aten.exp, aten.mul, aten.sigmoid, aten.slice_scatter]
# add_23 => add_141
# clone => clone_1
# contiguous => clone
# exp => exp
# imul => mul_237, slice_scatter_2
# mul => mul_236
# setitem => copy, slice_scatter
# setitem_1 => copy_1, slice_scatter_1
# sigmoid => sigmoid
triton_poi_fused_add_clone_copy_exp_mul_sigmoid_slice_scatter_11 = async_compile.triton('triton_poi_fused_add_clone_copy_exp_mul_sigmoid_slice_scatter_11', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_copy_exp_mul_sigmoid_slice_scatter_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_clone_copy_exp_mul_sigmoid_slice_scatter_11(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 391680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 85
    x4 = xindex
    x2 = (xindex // 255) % 192
    x1 = (xindex // 85) % 3
    tmp49 = tl.load(in_ptr0 + (x4), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 4, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp4 & tmp2
    tmp6 = tmp5 & tmp2
    tmp7 = tmp0 < tmp3
    tmp8 = tmp7 & tmp6
    tmp9 = tl.load(in_ptr0 + (x4), tmp8 & xmask, other=0.0)
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tl.load(in_ptr1 + (x0 + (2*x2)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.where(tmp8, tmp12, 0.0)
    tmp14 = tl.load(in_ptr0 + (x4), tmp6 & xmask, other=0.0)
    tmp15 = tl.where(tmp7, tmp13, tmp14)
    tmp16 = tl.exp(tmp15)
    tmp17 = tl.load(in_ptr2 + ((-2) + x0 + (2*x1)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.where(tmp6, tmp18, 0.0)
    tmp20 = tmp7 & tmp2
    tmp21 = tl.load(in_ptr0 + (x4), tmp20 & xmask, other=0.0)
    tmp22 = tl.sigmoid(tmp21)
    tmp23 = tl.load(in_ptr1 + (x0 + (2*x2)), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.where(tmp20, tmp24, 0.0)
    tmp26 = tl.load(in_ptr0 + (x4), tmp2 & xmask, other=0.0)
    tmp27 = tl.where(tmp7, tmp25, tmp26)
    tmp28 = tl.where(tmp5, tmp19, tmp27)
    tmp29 = 32.0
    tmp30 = tmp28 * tmp29
    tmp31 = tl.where(tmp2, tmp30, 0.0)
    tmp32 = tmp7 & tmp5
    tmp33 = tl.load(in_ptr0 + (x4), tmp32 & xmask, other=0.0)
    tmp34 = tl.sigmoid(tmp33)
    tmp35 = tl.load(in_ptr1 + (x0 + (2*x2)), tmp32 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp34 + tmp35
    tmp37 = tl.where(tmp32, tmp36, 0.0)
    tmp38 = tl.load(in_ptr0 + (x4), tmp5 & xmask, other=0.0)
    tmp39 = tl.where(tmp7, tmp37, tmp38)
    tmp40 = tl.exp(tmp39)
    tmp41 = tl.load(in_ptr2 + ((-2) + x0 + (2*x1)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp40 * tmp41
    tmp43 = tl.where(tmp5, tmp42, 0.0)
    tmp44 = tl.load(in_ptr0 + (x4), tmp7 & xmask, other=0.0)
    tmp45 = tl.sigmoid(tmp44)
    tmp46 = tl.load(in_ptr1 + (x0 + (2*x2)), tmp7 & xmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tmp45 + tmp46
    tmp48 = tl.where(tmp7, tmp47, 0.0)
    tmp50 = tl.where(tmp7, tmp48, tmp49)
    tmp51 = tl.where(tmp5, tmp43, tmp50)
    tmp52 = tl.where(tmp2, tmp31, tmp51)
    tl.store(out_ptr0 + (x4), tmp52, xmask)
''')
