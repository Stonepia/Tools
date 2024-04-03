

# Original file: ./yolov3___60.0/yolov3___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/wd/cwd53vxlwp5lv6yiuobua3ov4dxzyld6333nfawfa7ifp5ingrfe.py
# Source Nodes: [add_25, clone_2, contiguous_2, exp_2, imul_2, mul_2, setitem_6, setitem_7, sigmoid_2], Original ATen: [aten.add, aten.clone, aten.copy, aten.exp, aten.mul, aten.sigmoid, aten.slice_scatter]
# add_25 => add_175
# clone_2 => clone_5
# contiguous_2 => clone_4
# exp_2 => exp_2
# imul_2 => mul_305, slice_scatter_12
# mul_2 => mul_304
# setitem_6 => copy_6, slice_scatter_10
# setitem_7 => copy_7, slice_scatter_11
# sigmoid_2 => sigmoid_4
triton_poi_fused_add_clone_copy_exp_mul_sigmoid_slice_scatter_25 = async_compile.triton('triton_poi_fused_add_clone_copy_exp_mul_sigmoid_slice_scatter_25', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_copy_exp_mul_sigmoid_slice_scatter_25', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_clone_copy_exp_mul_sigmoid_slice_scatter_25(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6266880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 85
    x4 = xindex
    x2 = (xindex // 255) % 3072
    x1 = (xindex // 85) % 3
    tmp61 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 4, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp4 & tmp2
    tmp6 = tmp5 & tmp2
    tmp7 = tmp0 < tmp3
    tmp8 = tmp7 & tmp6
    tmp9 = tl.load(in_ptr0 + (x4), tmp8, other=0.0).to(tl.float32)
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.load(in_ptr1 + (x0 + (2*x2)), tmp8, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tl.where(tmp8, tmp14, 0.0)
    tmp16 = tl.load(in_ptr0 + (x4), tmp6, other=0.0).to(tl.float32)
    tmp17 = tl.where(tmp7, tmp15, tmp16)
    tmp18 = tl.exp(tmp17)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.load(in_ptr2 + ((-2) + x0 + (2*x1)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tl.where(tmp6, tmp22, 0.0)
    tmp24 = tmp7 & tmp2
    tmp25 = tl.load(in_ptr0 + (x4), tmp24, other=0.0).to(tl.float32)
    tmp26 = tl.sigmoid(tmp25)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tl.load(in_ptr1 + (x0 + (2*x2)), tmp24, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tl.where(tmp24, tmp30, 0.0)
    tmp32 = tl.load(in_ptr0 + (x4), tmp2, other=0.0).to(tl.float32)
    tmp33 = tl.where(tmp7, tmp31, tmp32)
    tmp34 = tl.where(tmp5, tmp23, tmp33)
    tmp35 = 8.0
    tmp36 = tmp34 * tmp35
    tmp37 = tl.where(tmp2, tmp36, 0.0)
    tmp38 = tmp7 & tmp5
    tmp39 = tl.load(in_ptr0 + (x4), tmp38, other=0.0).to(tl.float32)
    tmp40 = tl.sigmoid(tmp39)
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tl.load(in_ptr1 + (x0 + (2*x2)), tmp38, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tl.where(tmp38, tmp44, 0.0)
    tmp46 = tl.load(in_ptr0 + (x4), tmp5, other=0.0).to(tl.float32)
    tmp47 = tl.where(tmp7, tmp45, tmp46)
    tmp48 = tl.exp(tmp47)
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tl.load(in_ptr2 + ((-2) + x0 + (2*x1)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp49 * tmp50
    tmp52 = tmp51.to(tl.float32)
    tmp53 = tl.where(tmp5, tmp52, 0.0)
    tmp54 = tl.load(in_ptr0 + (x4), tmp7, other=0.0).to(tl.float32)
    tmp55 = tl.sigmoid(tmp54)
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tl.load(in_ptr1 + (x0 + (2*x2)), tmp7, eviction_policy='evict_last', other=0.0)
    tmp58 = tmp56 + tmp57
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tl.where(tmp7, tmp59, 0.0)
    tmp62 = tl.where(tmp7, tmp60, tmp61)
    tmp63 = tl.where(tmp5, tmp53, tmp62)
    tmp64 = tl.where(tmp2, tmp37, tmp63)
    tl.store(out_ptr0 + (x4), tmp64, None)
''')
