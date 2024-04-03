

# Original file: ./XLNetLMHeadModel__0_backward_567.1/XLNetLMHeadModel__0_backward_567.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/tv/ctvz37aebcwpgqznhy7tljzkbwadn4ibv5emfzebxyzcadycfggk.py
# Source Nodes: [gelu_23], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
# gelu_23 => add_260, erf_23, mul_382
triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8 = async_compile.triton('triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp6 = tl.load(in_ptr1 + (x0), None)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = 0.7071067811865476
    tmp8 = tmp6 * tmp7
    tmp9 = libdevice.erf(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = 0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tmp6 * tmp6
    tmp15 = -0.5
    tmp16 = tmp14 * tmp15
    tmp17 = tl.exp(tmp16)
    tmp18 = 0.3989422804014327
    tmp19 = tmp17 * tmp18
    tmp20 = tmp6 * tmp19
    tmp21 = tmp13 + tmp20
    tmp22 = tmp5 * tmp21
    tl.store(in_out_ptr0 + (x0), tmp22, None)
''')
