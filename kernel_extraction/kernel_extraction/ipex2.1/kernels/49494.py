

# Original file: ./xcit_large_24_p8_224___60.0/xcit_large_24_p8_224___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/ix/cixmohm7ingy66kyrl7kaygdmvogbxvslf4b6e4qqitdrjc3aqil.py
# Source Nodes: [stack], Original ATen: [aten.stack]
# stack => cat
triton_poi_fused_stack_6 = async_compile.triton('triton_poi_fused_stack_6', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_6', 'configs': [AttrsDescriptor(divisible_by_16=(1,), equal_to_1=())]})
@triton.jit
def triton_poi_fused_stack_6(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16) % 28
    x0 = xindex % 16
    x4 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3 + tmp2
    tmp5 = 28.000001907348633
    tmp6 = tmp4 / tmp5
    tmp7 = 6.283185307179586
    tmp8 = tmp6 * tmp7
    tmp9 = 1 + (2*x0)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp2
    tmp12 = 0.0
    tmp13 = tmp11 + tmp12
    tmp14 = 2.0
    tmp15 = tmp13 / tmp14
    tmp16 = libdevice.floor(tmp15)
    tmp17 = tmp16 * tmp14
    tmp18 = 32.0
    tmp19 = tmp17 / tmp18
    tmp20 = 10000.0
    tmp21 = libdevice.pow(tmp20, tmp19)
    tmp22 = tmp8 / tmp21
    tmp23 = tl.cos(tmp22)
    tl.store(out_ptr0 + (2*x4), tmp23, xmask)
''')
