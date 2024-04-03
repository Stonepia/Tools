

# Original file: ./ese_vovnet19b_dw___60.0/ese_vovnet19b_dw___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/yf/cyfn73dgb72ebx266hfcf2nu4ehiltd5b36bbiqejeztcootecqi.py
# Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___attn_gate, getattr_l__mod___stages___1___pool, mul], Original ATen: [aten.hardsigmoid, aten.max_pool2d_with_indices, aten.mul]
# getattr_getattr_l__mod___stages___0___blocks___0___attn_gate => add_16, clamp_max, clamp_min, convert_element_type_24, convert_element_type_25, div
# getattr_l__mod___stages___1___pool => max_pool2d_with_indices
# mul => mul_24
triton_poi_fused_hardsigmoid_max_pool2d_with_indices_mul_5 = async_compile.triton('triton_poi_fused_hardsigmoid_max_pool2d_with_indices_mul_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardsigmoid_max_pool2d_with_indices_mul_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_hardsigmoid_max_pool2d_with_indices_mul_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 42467328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 9216) % 36
    x1 = (xindex // 256) % 36
    x0 = xindex % 256
    x5 = (xindex // 9216)
    x6 = xindex
    tmp0 = 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 72, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (x0 + (512*x1) + (36864*x5)), tmp10, other=0.0).to(tl.float32)
    tmp12 = tl.where(tmp10, tmp11, float("-inf"))
    tmp13 = 1 + (2*x1)
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp3
    tmp16 = tmp14 & tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = tl.load(in_ptr0 + (256 + x0 + (512*x1) + (36864*x5)), tmp17, other=0.0).to(tl.float32)
    tmp19 = tl.where(tmp17, tmp18, float("-inf"))
    tmp20 = triton_helpers.maximum(tmp19, tmp12)
    tmp21 = 2 + (2*x1)
    tmp22 = tmp21 >= tmp1
    tmp23 = tmp21 < tmp3
    tmp24 = tmp22 & tmp23
    tmp25 = tmp5 & tmp24
    tmp26 = tl.load(in_ptr0 + (512 + x0 + (512*x1) + (36864*x5)), tmp25, other=0.0).to(tl.float32)
    tmp27 = tl.where(tmp25, tmp26, float("-inf"))
    tmp28 = triton_helpers.maximum(tmp27, tmp20)
    tmp29 = 1 + (2*x2)
    tmp30 = tmp29 >= tmp1
    tmp31 = tmp29 < tmp3
    tmp32 = tmp30 & tmp31
    tmp33 = tmp32 & tmp9
    tmp34 = tl.load(in_ptr0 + (18432 + x0 + (512*x1) + (36864*x5)), tmp33, other=0.0).to(tl.float32)
    tmp35 = tl.where(tmp33, tmp34, float("-inf"))
    tmp36 = triton_helpers.maximum(tmp35, tmp28)
    tmp37 = tmp32 & tmp16
    tmp38 = tl.load(in_ptr0 + (18688 + x0 + (512*x1) + (36864*x5)), tmp37, other=0.0).to(tl.float32)
    tmp39 = tl.where(tmp37, tmp38, float("-inf"))
    tmp40 = triton_helpers.maximum(tmp39, tmp36)
    tmp41 = tmp32 & tmp24
    tmp42 = tl.load(in_ptr0 + (18944 + x0 + (512*x1) + (36864*x5)), tmp41, other=0.0).to(tl.float32)
    tmp43 = tl.where(tmp41, tmp42, float("-inf"))
    tmp44 = triton_helpers.maximum(tmp43, tmp40)
    tmp45 = 2 + (2*x2)
    tmp46 = tmp45 >= tmp1
    tmp47 = tmp45 < tmp3
    tmp48 = tmp46 & tmp47
    tmp49 = tmp48 & tmp9
    tmp50 = tl.load(in_ptr0 + (36864 + x0 + (512*x1) + (36864*x5)), tmp49, other=0.0).to(tl.float32)
    tmp51 = tl.where(tmp49, tmp50, float("-inf"))
    tmp52 = triton_helpers.maximum(tmp51, tmp44)
    tmp53 = tmp48 & tmp16
    tmp54 = tl.load(in_ptr0 + (37120 + x0 + (512*x1) + (36864*x5)), tmp53, other=0.0).to(tl.float32)
    tmp55 = tl.where(tmp53, tmp54, float("-inf"))
    tmp56 = triton_helpers.maximum(tmp55, tmp52)
    tmp57 = tmp48 & tmp24
    tmp58 = tl.load(in_ptr0 + (37376 + x0 + (512*x1) + (36864*x5)), tmp57, other=0.0).to(tl.float32)
    tmp59 = tl.where(tmp57, tmp58, float("-inf"))
    tmp60 = triton_helpers.maximum(tmp59, tmp56)
    tl.store(out_ptr0 + (x6), tmp60, None)
''')
