

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/mj/cmjc2hx2nicnjcdtr3mprfabxfad2xar3y4mlucsal3esi6wfab7.py
# Source Nodes: [abs_6, add_19, add_20, mul_52, mul_56, mul_57, neg_9, setitem_15, tanh_3, truediv_13], Original ATen: [aten.abs, aten.add, aten.copy, aten.div, aten.mul, aten.neg, aten.select_scatter, aten.tanh]
# abs_6 => abs_6
# add_19 => add_22
# add_20 => add_23
# mul_52 => mul_52
# mul_56 => mul_56
# mul_57 => mul_57
# neg_9 => neg_9
# setitem_15 => copy_15, select_scatter_7
# tanh_3 => tanh_3
# truediv_13 => div_13
triton_poi_fused_abs_add_copy_div_mul_neg_select_scatter_tanh_13 = async_compile.triton('triton_poi_fused_abs_add_copy_div_mul_neg_select_scatter_tanh_13', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp16', 4: '*fp16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_mul_neg_select_scatter_tanh_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_mul_neg_select_scatter_tanh_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2090400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x4 = (xindex // 2)
    x3 = (xindex // 10400)
    x5 = (xindex // 2) % 5200
    x2 = (xindex // 52) % 200
    x8 = xindex
    tmp3 = tl.load(in_ptr0 + (x4), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (5356 + x5 + (5304*x3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp54 = tl.load(in_ptr3 + (21426 + x0 + (4*x5) + (21216*x3)), xmask).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.abs(tmp3)
    tmp5 = -tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 / tmp6
    tmp9 = libdevice.tanh(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = 0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13 * tmp3
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = 1 + x3
    tmp20 = tl.full([1], 1, tl.int64)
    tmp21 = tmp19 >= tmp20
    tmp22 = tl.full([1], 202, tl.int64)
    tmp23 = tmp19 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = 2 + x2
    tmp26 = tl.full([1], 2, tl.int64)
    tmp27 = tmp25 >= tmp26
    tmp28 = tmp25 < tmp22
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp24
    tmp31 = tl.full([1], 0, tl.int32)
    tmp32 = tmp1 == tmp31
    tmp33 = tl.load(in_ptr2 + (x4), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.abs(tmp33)
    tmp35 = -tmp34
    tmp36 = tmp35 + tmp6
    tmp37 = tmp36 / tmp6
    tmp38 = libdevice.tanh(tmp37)
    tmp39 = tmp38 + tmp10
    tmp40 = tmp39 * tmp12
    tmp41 = tmp40 * tmp33
    tmp42 = tl.load(in_ptr1 + (5356 + x5 + (5304*x3)), tmp30 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp41 * tmp43
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tl.load(in_ptr3 + (21424 + x0 + (4*x5) + (21216*x3)), tmp30 & xmask, other=0.0).to(tl.float32)
    tmp47 = tl.where(tmp2, tmp45, tmp46)
    tmp48 = tl.load(in_ptr3 + (21426 + x0 + (4*x5) + (21216*x3)), tmp30 & xmask, other=0.0).to(tl.float32)
    tmp49 = tl.where(tmp32, tmp47, tmp48)
    tmp50 = tl.where(tmp30, tmp49, 0.0)
    tmp51 = tl.load(in_ptr3 + (21426 + x0 + (4*x5) + (21216*x3)), tmp24 & xmask, other=0.0).to(tl.float32)
    tmp52 = tl.where(tmp29, tmp50, tmp51)
    tmp53 = tl.where(tmp24, tmp52, 0.0)
    tmp55 = tl.where(tmp24, tmp53, tmp54)
    tmp56 = tl.where(tmp2, tmp18, tmp55)
    tl.store(out_ptr0 + (x8), tmp56, xmask)
''')
