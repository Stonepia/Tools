

# Original file: ./DebertaForMaskedLM__0_forward_133.0/DebertaForMaskedLM__0_forward_133.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/66/c66pcu7wkobq7xtcfwk3lbx6b2eejz56hewb7npfprxoz4ufb2q6.py
# Source Nodes: [l__mod___deberta_encoder_layer_0_attention_output_dense], Original ATen: [aten.view]
# l__mod___deberta_encoder_layer_0_attention_output_dense => view_12
triton_poi_fused_view_6 = async_compile.triton('triton_poi_fused_view_6', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_view_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(x1 % 512)) + (32768*(x0 // 64)) + (393216*(x1 // 512)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')