

# Original file: ./MobileBertForMaskedLM__0_forward_352.0/MobileBertForMaskedLM__0_forward_352.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/aw/cawyuxiubyiqxcknt72a6yq6df6sf35nudsdqfm3s5y5kmddq3k2.py
# Source Nodes: [add_46, add_47, l__mod___mobilebert_encoder_layer_2_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_3_bottleneck_input_dense, mul_25], Original ATen: [aten.add, aten.clone, aten.mul, aten.view]
# add_46 => add_46
# add_47 => add_47
# l__mod___mobilebert_encoder_layer_2_output_bottleneck_dropout => clone_15
# l__mod___mobilebert_encoder_layer_3_bottleneck_input_dense => view_122
# mul_25 => mul_31
triton_poi_fused_add_clone_mul_view_16 = async_compile.triton('triton_poi_fused_add_clone_mul_view_16', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_mul_view_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_clone_mul_view_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x2), tmp6, None)
''')
