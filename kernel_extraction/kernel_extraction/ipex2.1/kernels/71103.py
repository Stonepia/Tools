

# Original file: ./timm_nfnet___60.0/timm_nfnet___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/2s/c2sjgyt4taom2unq47yjywm7kykmcave3yjxuvq2nghdvawusjbt.py
# Source Nodes: [gelu_43, mul_88, mul__53], Original ATen: [aten.gelu, aten.mul]
# gelu_43 => add_101, convert_element_type_182, convert_element_type_183, erf_43, mul_366, mul_367, mul_368
# mul_88 => mul_370
# mul__53 => mul_369
triton_poi_fused_gelu_mul_30 = async_compile.triton('triton_poi_fused_gelu_mul_30', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_mul_30', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_gelu_mul_30(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7077888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp1 * tmp4
    tmp6 = libdevice.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = tmp3 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = 1.7015043497085571
    tmp12 = tmp10 * tmp11
    tmp13 = 0.9805806756909201
    tmp14 = tmp12 * tmp13
    tl.store(out_ptr0 + (x0), tmp14, None)
''')