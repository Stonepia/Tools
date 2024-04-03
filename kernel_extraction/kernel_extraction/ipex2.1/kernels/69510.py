

# Original file: ./poolformer_m36___60.0/poolformer_m36___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/mq/cmqmnbp3dvgjjfbtioilnirwrlh5v7kga7iy5mg53q5cphoibvoi.py
# Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___token_mixer_pool], Original ATen: [aten.avg_pool2d]
# getattr_getattr_l__mod___stages___1___blocks___0___token_mixer_pool => avg_pool2d_6
triton_poi_fused_avg_pool2d_19 = async_compile.triton('triton_poi_fused_avg_pool2d_19', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_19', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_avg_pool2d_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9633792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5376) % 28
    x1 = (xindex // 192) % 28
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-5568) + x6), tmp10, other=0.0).to(tl.float32)
    tmp12 = tl.where(tmp10, tmp11, 0.0)
    tmp13 = x1
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp3
    tmp16 = tmp14 & tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = tl.load(in_ptr0 + ((-5376) + x6), tmp17, other=0.0).to(tl.float32)
    tmp19 = tl.where(tmp17, tmp18, 0.0)
    tmp20 = tmp19 + tmp12
    tmp21 = 1 + x1
    tmp22 = tmp21 >= tmp1
    tmp23 = tmp21 < tmp3
    tmp24 = tmp22 & tmp23
    tmp25 = tmp5 & tmp24
    tmp26 = tl.load(in_ptr0 + ((-5184) + x6), tmp25, other=0.0).to(tl.float32)
    tmp27 = tl.where(tmp25, tmp26, 0.0)
    tmp28 = tmp27 + tmp20
    tmp29 = x2
    tmp30 = tmp29 >= tmp1
    tmp31 = tmp29 < tmp3
    tmp32 = tmp30 & tmp31
    tmp33 = tmp32 & tmp9
    tmp34 = tl.load(in_ptr0 + ((-192) + x6), tmp33, other=0.0).to(tl.float32)
    tmp35 = tl.where(tmp33, tmp34, 0.0)
    tmp36 = tmp35 + tmp28
    tmp37 = tmp32 & tmp16
    tmp38 = tl.load(in_ptr0 + (x6), tmp37, other=0.0).to(tl.float32)
    tmp39 = tl.where(tmp37, tmp38, 0.0)
    tmp40 = tmp39 + tmp36
    tmp41 = tmp32 & tmp24
    tmp42 = tl.load(in_ptr0 + (192 + x6), tmp41, other=0.0).to(tl.float32)
    tmp43 = tl.where(tmp41, tmp42, 0.0)
    tmp44 = tmp43 + tmp40
    tmp45 = 1 + x2
    tmp46 = tmp45 >= tmp1
    tmp47 = tmp45 < tmp3
    tmp48 = tmp46 & tmp47
    tmp49 = tmp48 & tmp9
    tmp50 = tl.load(in_ptr0 + (5184 + x6), tmp49, other=0.0).to(tl.float32)
    tmp51 = tl.where(tmp49, tmp50, 0.0)
    tmp52 = tmp51 + tmp44
    tmp53 = tmp48 & tmp16
    tmp54 = tl.load(in_ptr0 + (5376 + x6), tmp53, other=0.0).to(tl.float32)
    tmp55 = tl.where(tmp53, tmp54, 0.0)
    tmp56 = tmp55 + tmp52
    tmp57 = tmp48 & tmp24
    tmp58 = tl.load(in_ptr0 + (5568 + x6), tmp57, other=0.0).to(tl.float32)
    tmp59 = tl.where(tmp57, tmp58, 0.0)
    tmp60 = tmp59 + tmp56
    tmp61 = 1.0
    tmp62 = tl.where(tmp10, tmp61, 0.0)
    tmp63 = tl.where(tmp17, tmp61, 0.0)
    tmp64 = tmp63 + tmp62
    tmp65 = tl.where(tmp25, tmp61, 0.0)
    tmp66 = tmp65 + tmp64
    tmp67 = tl.where(tmp33, tmp61, 0.0)
    tmp68 = tmp67 + tmp66
    tmp69 = tl.where(tmp37, tmp61, 0.0)
    tmp70 = tmp69 + tmp68
    tmp71 = tl.where(tmp41, tmp61, 0.0)
    tmp72 = tmp71 + tmp70
    tmp73 = tl.where(tmp49, tmp61, 0.0)
    tmp74 = tmp73 + tmp72
    tmp75 = tl.where(tmp53, tmp61, 0.0)
    tmp76 = tmp75 + tmp74
    tmp77 = tl.where(tmp57, tmp61, 0.0)
    tmp78 = tmp77 + tmp76
    tmp79 = tmp60 / tmp78
    tl.store(out_ptr0 + (x6), tmp79, None)
''')
