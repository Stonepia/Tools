

# Original file: ./cm3leon_generate__23_inference_63.3/cm3leon_generate__23_inference_63.3_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/cp/ccpyzyziaf35377ayk5dfyfdyvl2g6pj7z2sqxjbpiqboddbieqc.py
# Source Nodes: [iadd_7, nan_to_num, nan_to_num_6, softmax_6, triu], Original ATen: [aten._softmax, aten.add, aten.nan_to_num, aten.triu]
# iadd_7 => add_46
# nan_to_num => full_default_3, full_default_4
# nan_to_num_6 => eq_12, eq_13, isnan_6, where_19, where_20, where_21
# softmax_6 => amax_6, exp_6, sub_20, sum_7
# triu => full_default_1
triton_poi_fused__softmax_add_nan_to_num_triu_33 = async_compile.triton('triton_poi_fused__softmax_add_nan_to_num_triu_33', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_nan_to_num_triu_33', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__softmax_add_nan_to_num_triu_33(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (7*x0), xmask)
    tmp12 = tl.load(in_ptr1 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp15 = tl.load(in_ptr0 + (1 + (7*x0)), xmask)
    tmp24 = tl.load(in_ptr0 + (2 + (7*x0)), xmask)
    tmp33 = tl.load(in_ptr0 + (3 + (7*x0)), xmask)
    tmp42 = tl.load(in_ptr0 + (4 + (7*x0)), xmask)
    tmp51 = tl.load(in_ptr0 + (5 + (7*x0)), xmask)
    tmp60 = tl.load(in_ptr0 + (6 + (7*x0)), xmask)
    tmp1 = float("inf")
    tmp2 = tmp0 == tmp1
    tmp3 = float("-inf")
    tmp4 = tmp0 == tmp3
    tmp5 = libdevice.isnan(tmp0).to(tl.int1)
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = -3.4028234663852886e+38
    tmp9 = tl.where(tmp4, tmp8, tmp7)
    tmp10 = 3.4028234663852886e+38
    tmp11 = tl.where(tmp2, tmp10, tmp9)
    tmp14 = tmp11 + tmp13
    tmp16 = tmp15 == tmp1
    tmp17 = tmp15 == tmp3
    tmp18 = libdevice.isnan(tmp15).to(tl.int1)
    tmp19 = tl.where(tmp18, tmp6, tmp15)
    tmp20 = tl.where(tmp17, tmp8, tmp19)
    tmp21 = tl.where(tmp16, tmp10, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = triton_helpers.maximum(tmp14, tmp22)
    tmp25 = tmp24 == tmp1
    tmp26 = tmp24 == tmp3
    tmp27 = libdevice.isnan(tmp24).to(tl.int1)
    tmp28 = tl.where(tmp27, tmp6, tmp24)
    tmp29 = tl.where(tmp26, tmp8, tmp28)
    tmp30 = tl.where(tmp25, tmp10, tmp29)
    tmp31 = tmp30 + tmp13
    tmp32 = triton_helpers.maximum(tmp23, tmp31)
    tmp34 = tmp33 == tmp1
    tmp35 = tmp33 == tmp3
    tmp36 = libdevice.isnan(tmp33).to(tl.int1)
    tmp37 = tl.where(tmp36, tmp6, tmp33)
    tmp38 = tl.where(tmp35, tmp8, tmp37)
    tmp39 = tl.where(tmp34, tmp10, tmp38)
    tmp40 = tmp39 + tmp13
    tmp41 = triton_helpers.maximum(tmp32, tmp40)
    tmp43 = tmp42 == tmp1
    tmp44 = tmp42 == tmp3
    tmp45 = libdevice.isnan(tmp42).to(tl.int1)
    tmp46 = tl.where(tmp45, tmp6, tmp42)
    tmp47 = tl.where(tmp44, tmp8, tmp46)
    tmp48 = tl.where(tmp43, tmp10, tmp47)
    tmp49 = tmp48 + tmp13
    tmp50 = triton_helpers.maximum(tmp41, tmp49)
    tmp52 = tmp51 == tmp1
    tmp53 = tmp51 == tmp3
    tmp54 = libdevice.isnan(tmp51).to(tl.int1)
    tmp55 = tl.where(tmp54, tmp6, tmp51)
    tmp56 = tl.where(tmp53, tmp8, tmp55)
    tmp57 = tl.where(tmp52, tmp10, tmp56)
    tmp58 = tmp57 + tmp13
    tmp59 = triton_helpers.maximum(tmp50, tmp58)
    tmp61 = tmp60 == tmp1
    tmp62 = tmp60 == tmp3
    tmp63 = libdevice.isnan(tmp60).to(tl.int1)
    tmp64 = tl.where(tmp63, tmp6, tmp60)
    tmp65 = tl.where(tmp62, tmp8, tmp64)
    tmp66 = tl.where(tmp61, tmp10, tmp65)
    tmp67 = tmp66 + tmp13
    tmp68 = triton_helpers.maximum(tmp59, tmp67)
    tmp69 = tmp14 - tmp68
    tmp70 = tl.exp(tmp69)
    tmp71 = tmp22 - tmp68
    tmp72 = tl.exp(tmp71)
    tmp73 = tmp70 + tmp72
    tmp74 = tmp31 - tmp68
    tmp75 = tl.exp(tmp74)
    tmp76 = tmp73 + tmp75
    tmp77 = tmp40 - tmp68
    tmp78 = tl.exp(tmp77)
    tmp79 = tmp76 + tmp78
    tmp80 = tmp49 - tmp68
    tmp81 = tl.exp(tmp80)
    tmp82 = tmp79 + tmp81
    tmp83 = tmp58 - tmp68
    tmp84 = tl.exp(tmp83)
    tmp85 = tmp82 + tmp84
    tmp86 = tmp67 - tmp68
    tmp87 = tl.exp(tmp86)
    tmp88 = tmp85 + tmp87
    tl.store(out_ptr0 + (x0), tmp68, xmask)
    tl.store(out_ptr1 + (x0), tmp88, xmask)
''')
