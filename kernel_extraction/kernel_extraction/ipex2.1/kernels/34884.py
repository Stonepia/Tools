

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/jf/cjf7xl7ylzovx5sdwlzzr2kocqjpku4wic6snl46hhyy5wfqrv6s.py
# Source Nodes: [add_24, iadd_9, mul_60, setitem_19], Original ATen: [aten.add, aten.copy, aten.mul, aten.select_scatter, aten.slice_scatter]
# add_24 => add_28
# iadd_9 => slice_scatter_145
# mul_60 => mul_60
# setitem_19 => copy_19, select_scatter_9, slice_scatter_81
triton_poi_fused_add_copy_mul_select_scatter_slice_scatter_7 = async_compile.triton('triton_poi_fused_add_copy_mul_select_scatter_slice_scatter_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy_mul_select_scatter_slice_scatter_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_copy_mul_select_scatter_slice_scatter_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1060800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 26) % 204
    x0 = xindex % 26
    x3 = (xindex // 26)
    x2 = (xindex // 5304)
    x5 = xindex
    x4 = xindex % 5304
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x0
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = tmp6 == tmp7
    tmp9 = tl.load(in_ptr0 + (10608 + (26*x3)), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr0 + (10634 + (26*x3)), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = 0.5
    tmp13 = tmp11 * tmp12
    tmp14 = 2 + x2
    tmp15 = tl.full([1], 2, tl.int64)
    tmp16 = tmp14 >= tmp15
    tmp17 = tmp14 < tmp3
    tmp18 = tmp16 & tmp17
    tmp19 = tmp18 & tmp5
    tmp20 = tmp5 & tmp19
    tmp21 = tmp6 >= tmp1
    tmp22 = tmp21 & tmp20
    tmp23 = tl.load(in_ptr0 + (10608 + x5), tmp22 & xmask, other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr0 + (10607 + x5), tmp22 & xmask, other=0.0).to(tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.load(in_ptr0 + (10634 + x5), tmp22 & xmask, other=0.0).to(tl.float32)
    tmp27 = tmp25 + tmp26
    tmp28 = tl.load(in_ptr0 + (10633 + x5), tmp22 & xmask, other=0.0).to(tl.float32)
    tmp29 = tmp27 + tmp28
    tmp30 = 0.25
    tmp31 = tmp29 * tmp30
    tmp32 = tl.where(tmp22, tmp31, 0.0)
    tmp33 = 0.0
    tmp34 = tl.where(tmp21, tmp32, tmp33)
    tmp35 = tl.where(tmp20, tmp34, 0.0)
    tmp36 = tl.where(tmp5, tmp35, tmp33)
    tmp37 = tl.where(tmp19, tmp36, 0.0)
    tmp38 = tl.where(tmp18, tmp37, tmp33)
    tmp39 = tl.where(tmp8, tmp13, tmp38)
    tmp40 = tl.where(tmp5, tmp39, 0.0)
    tmp41 = tmp5 & tmp18
    tmp42 = tmp21 & tmp41
    tmp43 = tl.load(in_ptr0 + (10608 + x5), tmp42 & xmask, other=0.0).to(tl.float32)
    tmp44 = tl.load(in_ptr0 + (10607 + x5), tmp42 & xmask, other=0.0).to(tl.float32)
    tmp45 = tmp43 + tmp44
    tmp46 = tl.load(in_ptr0 + (10634 + x5), tmp42 & xmask, other=0.0).to(tl.float32)
    tmp47 = tmp45 + tmp46
    tmp48 = tl.load(in_ptr0 + (10633 + x5), tmp42 & xmask, other=0.0).to(tl.float32)
    tmp49 = tmp47 + tmp48
    tmp50 = tmp49 * tmp30
    tmp51 = tl.where(tmp42, tmp50, 0.0)
    tmp52 = tl.where(tmp21, tmp51, tmp33)
    tmp53 = tl.where(tmp41, tmp52, 0.0)
    tmp54 = tl.where(tmp5, tmp53, tmp33)
    tmp55 = tl.where(tmp18, tmp54, 0.0)
    tmp56 = tl.where(tmp18, tmp55, tmp33)
    tmp57 = tl.where(tmp5, tmp40, tmp56)
    tmp58 = tmp0 >= tmp15
    tmp59 = tmp58 & tmp4
    tmp60 = tl.load(in_ptr1 + ((-52) + x4 + (5200*x2)), tmp59 & xmask, other=0.0).to(tl.float32)
    tmp61 = tl.where(tmp59, tmp60, 0.0)
    tmp62 = tmp59 & tmp18
    tmp63 = tl.full([1], 25, tl.int64)
    tmp64 = tmp6 < tmp63
    tmp65 = tmp64 & tmp62
    tmp66 = tl.load(in_ptr2 + (1 + x2), tmp65 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp67 = tl.load(in_ptr0 + (10608 + x5), tmp65 & xmask, other=0.0).to(tl.float32)
    tmp68 = tmp66 * tmp67
    tmp69 = tmp68.to(tl.float32)
    tmp70 = tl.load(in_ptr3 + ((-50) + x0 + (25*x1) + (5000*x2)), tmp65 & xmask, other=0.0)
    tmp71 = tl.abs(tmp70)
    tmp72 = -tmp71
    tmp73 = 0.001
    tmp74 = tmp72 + tmp73
    tmp75 = tmp74 / tmp73
    tmp76 = libdevice.tanh(tmp75)
    tmp77 = 1.0
    tmp78 = tmp76 + tmp77
    tmp79 = tmp78 * tmp12
    tmp80 = tmp69 * tmp79
    tmp81 = tmp70 * tmp70
    tmp82 = tmp80 * tmp81
    tmp83 = tl.load(in_ptr4 + (10608 + x5), tmp65 & xmask, other=0.0).to(tl.float32)
    tmp84 = tmp83.to(tl.float32)
    tmp85 = tmp82 * tmp84
    tmp86 = tmp33 + tmp85
    tmp87 = tmp86.to(tl.float32)
    tmp88 = tl.where(tmp65, tmp87, 0.0)
    tmp89 = tl.where(tmp64, tmp88, tmp33)
    tmp90 = tl.where(tmp62, tmp89, 0.0)
    tmp91 = tl.where(tmp59, tmp90, tmp33)
    tmp92 = tl.where(tmp18, tmp91, 0.0)
    tmp93 = tl.where(tmp18, tmp92, tmp33)
    tmp94 = tl.where(tmp59, tmp61, tmp93)
    tl.store(out_ptr0 + (x5), tmp57, xmask)
    tl.store(out_ptr1 + (x5), tmp94, xmask)
''')
