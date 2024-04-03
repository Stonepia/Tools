

# Original file: ./cm3leon_generate__23_inference_63.3/cm3leon_generate__23_inference_63.3_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/sq/csqgoudapkbvf7f7epp2s6fx26gqspa5fawrxedhljyqrzph6khr.py
# Source Nodes: [iadd_7, nan_to_num, nan_to_num_6, softmax_6], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.nan_to_num]
# iadd_7 => add_46
# nan_to_num => full_default_2, full_default_3, full_default_4
# nan_to_num_6 => convert_element_type_54, eq_12, eq_13, isnan_6, where_19, where_20, where_21
# softmax_6 => amax_6, convert_element_type_56, exp_6, sub_20, sum_7
triton_poi_fused__softmax__to_copy_add_nan_to_num_33 = async_compile.triton('triton_poi_fused__softmax__to_copy_add_nan_to_num_33', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__to_copy_add_nan_to_num_33', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__softmax__to_copy_add_nan_to_num_33(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (7*x0), xmask).to(tl.float32)
    tmp13 = tl.load(in_ptr1 + (0)).to(tl.float32)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp17 = tl.load(in_ptr0 + (1 + (7*x0)), xmask).to(tl.float32)
    tmp28 = tl.load(in_ptr0 + (2 + (7*x0)), xmask).to(tl.float32)
    tmp39 = tl.load(in_ptr0 + (3 + (7*x0)), xmask).to(tl.float32)
    tmp50 = tl.load(in_ptr0 + (4 + (7*x0)), xmask).to(tl.float32)
    tmp61 = tl.load(in_ptr0 + (5 + (7*x0)), xmask).to(tl.float32)
    tmp72 = tl.load(in_ptr0 + (6 + (7*x0)), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = float("inf")
    tmp3 = tmp1 == tmp2
    tmp4 = float("-inf")
    tmp5 = tmp1 == tmp4
    tmp6 = libdevice.isnan(tmp0).to(tl.int1)
    tmp7 = 0.0
    tmp8 = tl.where(tmp6, tmp7, tmp0)
    tmp9 = -65504.0
    tmp10 = tl.where(tmp5, tmp9, tmp8)
    tmp11 = 65504.0
    tmp12 = tl.where(tmp3, tmp11, tmp10)
    tmp15 = tmp12 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 == tmp2
    tmp20 = tmp18 == tmp4
    tmp21 = libdevice.isnan(tmp17).to(tl.int1)
    tmp22 = tl.where(tmp21, tmp7, tmp17)
    tmp23 = tl.where(tmp20, tmp9, tmp22)
    tmp24 = tl.where(tmp19, tmp11, tmp23)
    tmp25 = tmp24 + tmp14
    tmp26 = tmp25.to(tl.float32)
    tmp27 = triton_helpers.maximum(tmp16, tmp26)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 == tmp2
    tmp31 = tmp29 == tmp4
    tmp32 = libdevice.isnan(tmp28).to(tl.int1)
    tmp33 = tl.where(tmp32, tmp7, tmp28)
    tmp34 = tl.where(tmp31, tmp9, tmp33)
    tmp35 = tl.where(tmp30, tmp11, tmp34)
    tmp36 = tmp35 + tmp14
    tmp37 = tmp36.to(tl.float32)
    tmp38 = triton_helpers.maximum(tmp27, tmp37)
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp40 == tmp2
    tmp42 = tmp40 == tmp4
    tmp43 = libdevice.isnan(tmp39).to(tl.int1)
    tmp44 = tl.where(tmp43, tmp7, tmp39)
    tmp45 = tl.where(tmp42, tmp9, tmp44)
    tmp46 = tl.where(tmp41, tmp11, tmp45)
    tmp47 = tmp46 + tmp14
    tmp48 = tmp47.to(tl.float32)
    tmp49 = triton_helpers.maximum(tmp38, tmp48)
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tmp51 == tmp2
    tmp53 = tmp51 == tmp4
    tmp54 = libdevice.isnan(tmp50).to(tl.int1)
    tmp55 = tl.where(tmp54, tmp7, tmp50)
    tmp56 = tl.where(tmp53, tmp9, tmp55)
    tmp57 = tl.where(tmp52, tmp11, tmp56)
    tmp58 = tmp57 + tmp14
    tmp59 = tmp58.to(tl.float32)
    tmp60 = triton_helpers.maximum(tmp49, tmp59)
    tmp62 = tmp61.to(tl.float32)
    tmp63 = tmp62 == tmp2
    tmp64 = tmp62 == tmp4
    tmp65 = libdevice.isnan(tmp61).to(tl.int1)
    tmp66 = tl.where(tmp65, tmp7, tmp61)
    tmp67 = tl.where(tmp64, tmp9, tmp66)
    tmp68 = tl.where(tmp63, tmp11, tmp67)
    tmp69 = tmp68 + tmp14
    tmp70 = tmp69.to(tl.float32)
    tmp71 = triton_helpers.maximum(tmp60, tmp70)
    tmp73 = tmp72.to(tl.float32)
    tmp74 = tmp73 == tmp2
    tmp75 = tmp73 == tmp4
    tmp76 = libdevice.isnan(tmp72).to(tl.int1)
    tmp77 = tl.where(tmp76, tmp7, tmp72)
    tmp78 = tl.where(tmp75, tmp9, tmp77)
    tmp79 = tl.where(tmp74, tmp11, tmp78)
    tmp80 = tmp79 + tmp14
    tmp81 = tmp80.to(tl.float32)
    tmp82 = triton_helpers.maximum(tmp71, tmp81)
    tmp83 = tmp16 - tmp82
    tmp84 = tl.exp(tmp83)
    tmp85 = tmp26 - tmp82
    tmp86 = tl.exp(tmp85)
    tmp87 = tmp84 + tmp86
    tmp88 = tmp37 - tmp82
    tmp89 = tl.exp(tmp88)
    tmp90 = tmp87 + tmp89
    tmp91 = tmp48 - tmp82
    tmp92 = tl.exp(tmp91)
    tmp93 = tmp90 + tmp92
    tmp94 = tmp59 - tmp82
    tmp95 = tl.exp(tmp94)
    tmp96 = tmp93 + tmp95
    tmp97 = tmp70 - tmp82
    tmp98 = tl.exp(tmp97)
    tmp99 = tmp96 + tmp98
    tmp100 = tmp81 - tmp82
    tmp101 = tl.exp(tmp100)
    tmp102 = tmp99 + tmp101
    tl.store(out_ptr0 + (x0), tmp82, xmask)
    tl.store(out_ptr1 + (x0), tmp102, xmask)
''')
