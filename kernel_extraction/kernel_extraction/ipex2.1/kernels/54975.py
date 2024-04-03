

# Original file: ./MobileBertForMaskedLM__0_forward_352.0/MobileBertForMaskedLM__0_forward_352.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/2e/c2exjjge4t62k27wxmkcucowt3b4b6aqpm2pex45nhvvthsrky5t.py
# Source Nodes: [l__self___mobilebert_encoder_layer_0_ffn_0_intermediate_intermediate_act_fn, l__self___mobilebert_encoder_layer_0_ffn_0_output_dense], Original ATen: [aten.relu, aten.view]
# l__self___mobilebert_encoder_layer_0_ffn_0_intermediate_intermediate_act_fn => relu
# l__self___mobilebert_encoder_layer_0_ffn_0_output_dense => view_26
triton_poi_fused_relu_view_18 = async_compile.triton('triton_poi_fused_relu_view_18', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_view_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_relu_view_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tl.store(out_ptr0 + (x0), tmp1, None)
''')
