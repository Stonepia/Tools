

# Original file: ./dm_nfnet_f0___60.0/dm_nfnet_f0___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/dw/cdwdztpioywpmjd4yzfafp2z7dbes4qf67jiglyvkydbv5tbzyqq.py
# Source Nodes: [add_6, add_7, gelu_35, mul_60, mul_61, mul_62, mul_68, mul_69, mul_70, mul_71, mul__37, mul__42, mul__43], Original ATen: [aten.add, aten.gelu, aten.mul]
# add_6 => add_72
# add_7 => add_81
# gelu_35 => add_82, convert_element_type_181, convert_element_type_182, erf_35, mul_297, mul_298, mul_299
# mul_60 => mul_260
# mul_61 => mul_261
# mul_62 => mul_263
# mul_68 => mul_293
# mul_69 => mul_294
# mul_70 => mul_296
# mul_71 => mul_301
# mul__37 => mul_262
# mul__42 => mul_295
# mul__43 => mul_300
triton_poi_fused_add_gelu_mul_23 = async_compile.triton('triton_poi_fused_add_gelu_mul_23', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_mul_23', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_gelu_mul_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1536
    x2 = (xindex // 393216)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0 + (1536*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr1 + (x3), None).to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (x0 + (1536*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr3 + (x3), None).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = 3.1128385066986084
    tmp6 = tmp4 * tmp5
    tmp7 = 0.2
    tmp8 = tmp6 * tmp7
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11 * tmp3
    tmp13 = -0.888830304145813
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14 * tmp7
    tmp17 = tmp15 + tmp16
    tmp18 = tmp8 + tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = 0.5
    tmp21 = tmp19 * tmp20
    tmp22 = 0.7071067811865476
    tmp23 = tmp19 * tmp22
    tmp24 = libdevice.erf(tmp23)
    tmp25 = 1.0
    tmp26 = tmp24 + tmp25
    tmp27 = tmp21 * tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = 1.7015043497085571
    tmp30 = tmp28 * tmp29
    tmp31 = 0.9128709291752768
    tmp32 = tmp30 * tmp31
    tl.store(in_out_ptr0 + (x3), tmp18, None)
    tl.store(out_ptr0 + (x3), tmp32, None)
''')
