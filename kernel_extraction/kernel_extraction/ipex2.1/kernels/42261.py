

# Original file: ./nfnet_l0___60.0/nfnet_l0___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/us/cusyspqzdt6imgohygv3dvwn3vxxxgasrcxrgtnelpd7xerwgeru.py
# Source Nodes: [add_2, getattr_getattr_l__mod___stages___2_____0___act1, mul_27, mul_28, mul_29, mul_30], Original ATen: [aten.add, aten.mul, aten.silu]
# add_2 => add_20
# getattr_getattr_l__mod___stages___2_____0___act1 => convert_element_type_66, convert_element_type_67, mul_81, sigmoid_18
# mul_27 => mul_78
# mul_28 => mul_79
# mul_29 => mul_80
# mul_30 => mul_82
triton_poi_fused_add_mul_silu_13 = async_compile.triton('triton_poi_fused_add_mul_silu_13', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_silu_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_silu_13(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 84934656
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 512
    x2 = (xindex // 663552)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0 + (512*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x3), None).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.2
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 0.9622504486493761
    tmp14 = tmp12 * tmp13
    tl.store(in_out_ptr0 + (x3), tmp14, None)
''')
