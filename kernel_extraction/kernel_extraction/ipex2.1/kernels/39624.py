

# Original file: ./levit_128___60.0/levit_128___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/r3/cr35jxjvzfcowtabchb3rnavl7ztwiqh6ihiumeje3khaabmanyb.py
# Source Nodes: [getattr_getattr_l__self___stages___0___blocks___0___attn_proj_act], Original ATen: [aten.hardswish]
# getattr_getattr_l__self___stages___0___blocks___0___attn_proj_act => add_14, clamp_max_3, clamp_min_3, convert_element_type_28, convert_element_type_29, div_4, mul_19
triton_poi_fused_hardswish_9 = async_compile.triton('triton_poi_fused_hardswish_9', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_hardswish_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 196
    x2 = (xindex // 25088)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*x1) + (6272*(x0 // 32)) + (25088*x2) + (x0 % 32)), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 3.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = 6.0
    tmp7 = triton_helpers.minimum(tmp5, tmp6)
    tmp8 = tmp1 * tmp7
    tmp9 = tmp8 / tmp6
    tmp10 = tmp9.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp10, None)
''')
