

# Original file: ./DebertaV2ForMaskedLM__0_forward_205.0/DebertaV2ForMaskedLM__0_forward_205.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/mw/cmwyy4vkezhk4xplfswle5jcqxkeikkumexf5omsgj3twcymmuds.py
# Source Nodes: [to_1, truediv], Original ATen: [aten._to_copy, aten.div]
# to_1 => full_default_2
# truediv => div
triton_poi_fused__to_copy_div_5 = async_compile.triton('triton_poi_fused__to_copy_div_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_div_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_div_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 48
    x2 = (xindex // 3072)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*(x1 % 24)) + (1536*x2) + (786432*(x1 // 24))), None).to(tl.float32)
    tmp1 = 8.0
    tmp2 = tmp0 / tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')
