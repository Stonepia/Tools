

# Original file: ./adv_inception_v3___60.0/adv_inception_v3___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/qi/cqib46jmivj7sbav2xljstntckf3hdsdkpv7un2xt6ym4eq6hvnq.py
# Source Nodes: [avg_pool2d_1], Original ATen: [aten.avg_pool2d]
# avg_pool2d_1 => avg_pool2d_1
triton_poi_fused_avg_pool2d_7 = async_compile.triton('triton_poi_fused_avg_pool2d_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_avg_pool2d_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40140800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 8960) % 35
    x1 = (xindex // 256) % 35
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 35, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-9216) + x6), tmp10, other=0.0).to(tl.float32)
    tmp12 = tl.where(tmp10, tmp11, 0.0)
    tmp13 = x1
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp3
    tmp16 = tmp14 & tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = tl.load(in_ptr0 + ((-8960) + x6), tmp17, other=0.0).to(tl.float32)
    tmp19 = tl.where(tmp17, tmp18, 0.0)
    tmp20 = tmp19 + tmp12
    tmp21 = 1 + x1
    tmp22 = tmp21 >= tmp1
    tmp23 = tmp21 < tmp3
    tmp24 = tmp22 & tmp23
    tmp25 = tmp5 & tmp24
    tmp26 = tl.load(in_ptr0 + ((-8704) + x6), tmp25, other=0.0).to(tl.float32)
    tmp27 = tl.where(tmp25, tmp26, 0.0)
    tmp28 = tmp27 + tmp20
    tmp29 = x2
    tmp30 = tmp29 >= tmp1
    tmp31 = tmp29 < tmp3
    tmp32 = tmp30 & tmp31
    tmp33 = tmp32 & tmp9
    tmp34 = tl.load(in_ptr0 + ((-256) + x6), tmp33, other=0.0).to(tl.float32)
    tmp35 = tl.where(tmp33, tmp34, 0.0)
    tmp36 = tmp35 + tmp28
    tmp37 = tmp32 & tmp16
    tmp38 = tl.load(in_ptr0 + (x6), tmp37, other=0.0).to(tl.float32)
    tmp39 = tl.where(tmp37, tmp38, 0.0)
    tmp40 = tmp39 + tmp36
    tmp41 = tmp32 & tmp24
    tmp42 = tl.load(in_ptr0 + (256 + x6), tmp41, other=0.0).to(tl.float32)
    tmp43 = tl.where(tmp41, tmp42, 0.0)
    tmp44 = tmp43 + tmp40
    tmp45 = 1 + x2
    tmp46 = tmp45 >= tmp1
    tmp47 = tmp45 < tmp3
    tmp48 = tmp46 & tmp47
    tmp49 = tmp48 & tmp9
    tmp50 = tl.load(in_ptr0 + (8704 + x6), tmp49, other=0.0).to(tl.float32)
    tmp51 = tl.where(tmp49, tmp50, 0.0)
    tmp52 = tmp51 + tmp44
    tmp53 = tmp48 & tmp16
    tmp54 = tl.load(in_ptr0 + (8960 + x6), tmp53, other=0.0).to(tl.float32)
    tmp55 = tl.where(tmp53, tmp54, 0.0)
    tmp56 = tmp55 + tmp52
    tmp57 = tmp48 & tmp24
    tmp58 = tl.load(in_ptr0 + (9216 + x6), tmp57, other=0.0).to(tl.float32)
    tmp59 = tl.where(tmp57, tmp58, 0.0)
    tmp60 = tmp59 + tmp56
    tmp61 = tl.full([1], -1, tl.int64)
    tmp62 = tmp0 >= tmp61
    tmp63 = tl.full([1], 36, tl.int64)
    tmp64 = tmp0 < tmp63
    tmp65 = tmp62 & tmp64
    tmp66 = tmp6 >= tmp61
    tmp67 = tmp6 < tmp63
    tmp68 = tmp66 & tmp67
    tmp69 = tmp65 & tmp68
    tmp70 = tmp10 & tmp69
    tmp71 = 1.0
    tmp72 = tl.where(tmp70, tmp71, 1.0)
    tmp73 = tl.where(tmp69, tmp72, 0.0)
    tmp74 = tmp13 >= tmp61
    tmp75 = tmp13 < tmp63
    tmp76 = tmp74 & tmp75
    tmp77 = tmp65 & tmp76
    tmp78 = tmp17 & tmp77
    tmp79 = tl.where(tmp78, tmp71, 1.0)
    tmp80 = tl.where(tmp77, tmp79, 0.0)
    tmp81 = tmp80 + tmp73
    tmp82 = tmp21 >= tmp61
    tmp83 = tmp21 < tmp63
    tmp84 = tmp82 & tmp83
    tmp85 = tmp65 & tmp84
    tmp86 = tmp25 & tmp85
    tmp87 = tl.where(tmp86, tmp71, 1.0)
    tmp88 = tl.where(tmp85, tmp87, 0.0)
    tmp89 = tmp88 + tmp81
    tmp90 = tmp29 >= tmp61
    tmp91 = tmp29 < tmp63
    tmp92 = tmp90 & tmp91
    tmp93 = tmp92 & tmp68
    tmp94 = tmp33 & tmp93
    tmp95 = tl.where(tmp94, tmp71, 1.0)
    tmp96 = tl.where(tmp93, tmp95, 0.0)
    tmp97 = tmp96 + tmp89
    tmp98 = tmp92 & tmp76
    tmp99 = tmp37 & tmp98
    tmp100 = tl.where(tmp99, tmp71, 1.0)
    tmp101 = tl.where(tmp98, tmp100, 0.0)
    tmp102 = tmp101 + tmp97
    tmp103 = tmp92 & tmp84
    tmp104 = tmp41 & tmp103
    tmp105 = tl.where(tmp104, tmp71, 1.0)
    tmp106 = tl.where(tmp103, tmp105, 0.0)
    tmp107 = tmp106 + tmp102
    tmp108 = tmp45 >= tmp61
    tmp109 = tmp45 < tmp63
    tmp110 = tmp108 & tmp109
    tmp111 = tmp110 & tmp68
    tmp112 = tmp49 & tmp111
    tmp113 = tl.where(tmp112, tmp71, 1.0)
    tmp114 = tl.where(tmp111, tmp113, 0.0)
    tmp115 = tmp114 + tmp107
    tmp116 = tmp110 & tmp76
    tmp117 = tmp53 & tmp116
    tmp118 = tl.where(tmp117, tmp71, 1.0)
    tmp119 = tl.where(tmp116, tmp118, 0.0)
    tmp120 = tmp119 + tmp115
    tmp121 = tmp110 & tmp84
    tmp122 = tmp57 & tmp121
    tmp123 = tl.where(tmp122, tmp71, 1.0)
    tmp124 = tl.where(tmp121, tmp123, 0.0)
    tmp125 = tmp124 + tmp120
    tmp126 = tmp60 / tmp125
    tl.store(out_ptr0 + (x6), tmp126, None)
''')