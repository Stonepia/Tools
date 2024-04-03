

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/m6/cm6n22cxucqkt7xkrzwjhx3kuk6ns6mijmagsufgjt65ziarvo3q.py
# Source Nodes: [add_9, iadd_51, mul_100, mul_101, mul_98, mul_99, sub_28, sub_29, truediv_65, truediv_66], Original ATen: [aten.add, aten.div, aten.mul, aten.sub]
# add_9 => add_60
# iadd_51 => add_61
# mul_100 => mul_103
# mul_101 => mul_104
# mul_98 => mul_101
# mul_99 => mul_102
# sub_28 => sub_28
# sub_29 => sub_29
# truediv_65 => div_62
# truediv_66 => div_63
triton_poi_fused_add_div_mul_sub_64 = async_compile.triton('triton_poi_fused_add_div_mul_sub_64', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*bf16', 9: '*bf16', 10: '*bf16', 11: '*bf16', 12: '*bf16', 13: '*bf16', 14: '*bf16', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_sub_64', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_div_mul_sub_64(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 26) % 200
    x2 = (xindex // 5200)
    x3 = xindex % 5200
    x4 = xindex
    tmp33 = tl.load(in_ptr4 + (2 + x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp34 = tl.load(in_ptr5 + (2 + x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp66 = tl.load(in_ptr9 + (2 + x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp90 = tl.load(in_ptr12 + (31981 + (3*x3) + (15912*x2)), xmask).to(tl.float32)
    tmp93 = tl.load(in_ptr13 + (10660 + x3 + (5304*x2)), xmask).to(tl.float32)
    tmp0 = 2 + x1
    tmp1 = tl.full([1], 203, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 203, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (10608 + x3 + (5278*x2)), tmp4 & xmask, other=0.0).to(tl.float32)
    tmp6 = 2000.0
    tmp7 = tmp5 * tmp6
    tmp8 = tl.load(in_ptr1 + (2 + x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tmp7 / tmp8
    tmp10 = tl.load(in_ptr2 + (10660 + x3 + (5304*x2)), tmp4 & xmask, other=0.0).to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tl.load(in_ptr3 + (2 + x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.where(tmp4, tmp13, 0.0)
    tmp15 = 0.0
    tmp16 = tl.where(tmp4, tmp14, tmp15)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = 1 + x1
    tmp19 = tmp18 == tmp1
    tmp20 = tmp18 < tmp3
    tmp21 = tl.load(in_ptr0 + (10582 + x3 + (5278*x2)), tmp20 & xmask, other=0.0).to(tl.float32)
    tmp22 = tmp21 * tmp6
    tmp23 = tl.load(in_ptr1 + (1 + x1), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp24 = tmp22 / tmp23
    tmp25 = tl.load(in_ptr2 + (10634 + x3 + (5304*x2)), tmp20 & xmask, other=0.0).to(tl.float32)
    tmp26 = tmp24 * tmp25
    tmp27 = tl.load(in_ptr3 + (1 + x1), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp28 = tmp26 * tmp27
    tmp29 = tl.where(tmp20, tmp28, 0.0)
    tmp30 = tl.where(tmp20, tmp29, tmp15)
    tmp31 = tl.where(tmp19, tmp15, tmp30)
    tmp32 = tmp17 - tmp31
    tmp35 = tmp33 * tmp34
    tmp36 = tmp32 / tmp35
    tmp37 = 2 + x2
    tmp38 = tmp37 == tmp1
    tmp39 = tmp37 < tmp3
    tmp40 = tl.load(in_ptr6 + (10660 + x3 + (5304*x2)), tmp39 & xmask, other=0.0).to(tl.float32)
    tmp41 = tmp40 * tmp6
    tmp42 = tl.load(in_ptr4 + (2 + x1), tmp39 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp43 = tl.load(in_ptr7 + (2 + x2), tmp39 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp44 = tmp42 * tmp43
    tmp45 = tmp41 / tmp44
    tmp46 = tl.load(in_ptr8 + (10660 + x3 + (5304*x2)), tmp39 & xmask, other=0.0).to(tl.float32)
    tmp47 = tmp45 * tmp46
    tmp48 = tl.where(tmp39, tmp47, 0.0)
    tmp49 = tl.where(tmp39, tmp48, tmp15)
    tmp50 = tl.where(tmp38, tmp15, tmp49)
    tmp51 = 1 + x2
    tmp52 = tmp51 == tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tl.load(in_ptr6 + (5356 + x3 + (5304*x2)), tmp53 & xmask, other=0.0).to(tl.float32)
    tmp55 = tmp54 * tmp6
    tmp56 = tl.load(in_ptr4 + (2 + x1), tmp53 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp57 = tl.load(in_ptr7 + (1 + x2), tmp53 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp58 = tmp56 * tmp57
    tmp59 = tmp55 / tmp58
    tmp60 = tl.load(in_ptr8 + (5356 + x3 + (5304*x2)), tmp53 & xmask, other=0.0).to(tl.float32)
    tmp61 = tmp59 * tmp60
    tmp62 = tl.where(tmp53, tmp61, 0.0)
    tmp63 = tl.where(tmp53, tmp62, tmp15)
    tmp64 = tl.where(tmp52, tmp15, tmp63)
    tmp65 = tmp50 - tmp64
    tmp67 = tmp33 * tmp66
    tmp68 = tmp65 / tmp67
    tmp69 = tmp68 + tmp36
    tmp70 = tl.full([1], 2, tl.int64)
    tmp71 = tmp37 >= tmp70
    tmp72 = tl.full([1], 202, tl.int64)
    tmp73 = tmp37 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tl.load(in_ptr10 + (157 + (3*x3) + (15912*x2)), tmp74 & xmask, other=0.0).to(tl.float32)
    tmp76 = tl.where(tmp74, tmp75, 0.0)
    tmp77 = tmp0 >= tmp70
    tmp78 = tmp0 < tmp72
    tmp79 = tmp77 & tmp78
    tmp80 = tmp79 & tmp74
    tmp81 = tl.full([1], 1, tl.int32)
    tmp82 = tmp81 == tmp81
    tmp83 = tl.load(in_ptr11 + (x4), tmp80 & xmask, other=0.0).to(tl.float32)
    tmp84 = tl.load(in_ptr12 + (31981 + (3*x3) + (15912*x2)), tmp80 & xmask, other=0.0).to(tl.float32)
    tmp85 = tl.where(tmp82, tmp83, tmp84)
    tmp86 = tl.where(tmp80, tmp85, 0.0)
    tmp87 = tl.load(in_ptr12 + (31981 + (3*x3) + (15912*x2)), tmp74 & xmask, other=0.0).to(tl.float32)
    tmp88 = tl.where(tmp79, tmp86, tmp87)
    tmp89 = tl.where(tmp74, tmp88, 0.0)
    tmp91 = tl.where(tmp74, tmp89, tmp90)
    tmp92 = tl.where(tmp74, tmp76, tmp91)
    tmp94 = 1.0
    tmp95 = tmp93 * tmp94
    tmp96 = tmp95 * tmp69
    tmp97 = tmp92 + tmp96
    tl.store(in_out_ptr0 + (x4), tmp97, xmask)
''')
