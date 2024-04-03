

# Original file: ./torch_multimodal_clip___60.0/torch_multimodal_clip___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/lj/clj53tgjxq663yix7vrpbbfecioskj2ed3vnjhltrqzxjox4zdlc.py
# Source Nodes: [l__mod___encoder_b_encoder], Original ATen: [aten.mul, aten.sigmoid]
# l__mod___encoder_b_encoder => mul_106, mul_107, sigmoid_12
triton_poi_fused_mul_sigmoid_24 = async_compile.triton('triton_poi_fused_mul_sigmoid_24', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_mul_sigmoid_24(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5046272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 1.702
    tmp2 = tmp0 * tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp3 * tmp0
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''')