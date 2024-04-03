

# Original file: ./cm3leon_generate__23_inference_63.3/cm3leon_generate__23_inference_63.3_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/wo/cwoudbidgzc3a3fdtezuuejfq2pfloajrxf44evuzxdtgy7zb3dt.py
# Source Nodes: [iadd_5, nan_to_num, nan_to_num_4, softmax_4], Original ATen: [aten._softmax, aten.add, aten.nan_to_num]
# iadd_5 => add_32
# nan_to_num => full_default_2, full_default_3, full_default_4
# nan_to_num_4 => convert_element_type_96, eq_8, eq_9, isnan_4, where_13, where_14, where_15
# softmax_4 => amax_4, exp_4, sub_14, sum_5
triton_poi_fused__softmax_add_nan_to_num_25 = async_compile.triton('triton_poi_fused__softmax_add_nan_to_num_25', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_nan_to_num_25', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__softmax_add_nan_to_num_25(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (5*x0), xmask).to(tl.float32)
    tmp14 = tl.load(in_ptr1 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp17 = tl.load(in_ptr0 + (1 + (5*x0)), xmask).to(tl.float32)
    tmp28 = tl.load(in_ptr0 + (2 + (5*x0)), xmask).to(tl.float32)
    tmp39 = tl.load(in_ptr0 + (3 + (5*x0)), xmask).to(tl.float32)
    tmp50 = tl.load(in_ptr0 + (4 + (5*x0)), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = float("inf")
    tmp3 = tmp1 == tmp2
    tmp4 = float("-inf")
    tmp5 = tmp1 == tmp4
    tmp6 = libdevice.isnan(tmp0).to(tl.int1)
    tmp7 = 0.0
    tmp8 = tl.where(tmp6, tmp7, tmp0)
    tmp9 = -3.3895313892515355e+38
    tmp10 = tl.where(tmp5, tmp9, tmp8)
    tmp11 = 3.3895313892515355e+38
    tmp12 = tl.where(tmp3, tmp11, tmp10)
    tmp13 = tmp12.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 == tmp2
    tmp20 = tmp18 == tmp4
    tmp21 = libdevice.isnan(tmp17).to(tl.int1)
    tmp22 = tl.where(tmp21, tmp7, tmp17)
    tmp23 = tl.where(tmp20, tmp9, tmp22)
    tmp24 = tl.where(tmp19, tmp11, tmp23)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 + tmp15
    tmp27 = triton_helpers.maximum(tmp16, tmp26)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 == tmp2
    tmp31 = tmp29 == tmp4
    tmp32 = libdevice.isnan(tmp28).to(tl.int1)
    tmp33 = tl.where(tmp32, tmp7, tmp28)
    tmp34 = tl.where(tmp31, tmp9, tmp33)
    tmp35 = tl.where(tmp30, tmp11, tmp34)
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp36 + tmp15
    tmp38 = triton_helpers.maximum(tmp27, tmp37)
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp40 == tmp2
    tmp42 = tmp40 == tmp4
    tmp43 = libdevice.isnan(tmp39).to(tl.int1)
    tmp44 = tl.where(tmp43, tmp7, tmp39)
    tmp45 = tl.where(tmp42, tmp9, tmp44)
    tmp46 = tl.where(tmp41, tmp11, tmp45)
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp47 + tmp15
    tmp49 = triton_helpers.maximum(tmp38, tmp48)
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tmp51 == tmp2
    tmp53 = tmp51 == tmp4
    tmp54 = libdevice.isnan(tmp50).to(tl.int1)
    tmp55 = tl.where(tmp54, tmp7, tmp50)
    tmp56 = tl.where(tmp53, tmp9, tmp55)
    tmp57 = tl.where(tmp52, tmp11, tmp56)
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tmp58 + tmp15
    tmp60 = triton_helpers.maximum(tmp49, tmp59)
    tmp61 = tmp16 - tmp60
    tmp62 = tl.exp(tmp61)
    tmp63 = tmp26 - tmp60
    tmp64 = tl.exp(tmp63)
    tmp65 = tmp62 + tmp64
    tmp66 = tmp37 - tmp60
    tmp67 = tl.exp(tmp66)
    tmp68 = tmp65 + tmp67
    tmp69 = tmp48 - tmp60
    tmp70 = tl.exp(tmp69)
    tmp71 = tmp68 + tmp70
    tmp72 = tmp59 - tmp60
    tmp73 = tl.exp(tmp72)
    tmp74 = tmp71 + tmp73
    tl.store(out_ptr0 + (x0), tmp60, xmask)
    tl.store(out_ptr1 + (x0), tmp74, xmask)
''')
