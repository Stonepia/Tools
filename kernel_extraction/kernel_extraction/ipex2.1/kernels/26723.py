

# Original file: ./LayoutLMForSequenceClassification__0_backward_207.1/LayoutLMForSequenceClassification__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/7m/c7mn4qyutdifeenqbj5fzq5dhdb6jaciv467kao6zs3o3wpfkhqc.py
# Source Nodes: [l__mod___layoutlm_pooler_activation], Original ATen: [aten.native_dropout_backward, aten.tanh, aten.tanh_backward]
# l__mod___layoutlm_pooler_activation => tanh
triton_poi_fused_native_dropout_backward_tanh_tanh_backward_5 = async_compile.triton('triton_poi_fused_native_dropout_backward_tanh_tanh_backward_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_dropout_backward_tanh_tanh_backward_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_native_dropout_backward_tanh_tanh_backward_5(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
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
    tmp7 = libdevice.tanh(tmp6)
    tmp8 = tmp7 * tmp7
    tmp9 = 1.0
    tmp10 = tmp9 - tmp8
    tmp11 = tmp5 * tmp10
    tl.store(in_out_ptr0 + (x0), tmp11, None)
''')
