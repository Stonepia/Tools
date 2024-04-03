

# Original file: ./MobileBertForQuestionAnswering__0_forward_349.0/MobileBertForQuestionAnswering__0_forward_349.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/rh/crhzemy4vsjm3d6fslw7gyijlpi2gcswn6fyhh3cms76uxjdkuab.py
# Source Nodes: [l__mod___mobilebert_encoder_layer_0_ffn_0_intermediate_intermediate_act_fn, l__mod___mobilebert_encoder_layer_0_ffn_0_output_dense], Original ATen: [aten.relu, aten.view]
# l__mod___mobilebert_encoder_layer_0_ffn_0_intermediate_intermediate_act_fn => relu
# l__mod___mobilebert_encoder_layer_0_ffn_0_output_dense => view_26
triton_poi_fused_relu_view_11 = async_compile.triton('triton_poi_fused_relu_view_11', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_view_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_relu_view_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tl.store(out_ptr0 + (x0), tmp1, None)
''')
