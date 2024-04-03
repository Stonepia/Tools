

# Original file: ./MobileBertForQuestionAnswering__0_forward_349.0/MobileBertForQuestionAnswering__0_forward_349.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/fc/cfczdnl7peeh55qpam66ry2qbgepueut267yicrqy5zcmkyh5i5d.py
# Source Nodes: [l__mod___mobilebert_encoder_layer_0_attention_output_dense], Original ATen: [aten.view]
# l__mod___mobilebert_encoder_layer_0_attention_output_dense => view_22
triton_poi_fused_view_9 = async_compile.triton('triton_poi_fused_view_9', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_view_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*(x1 % 128)) + (4096*(x0 // 32)) + (16384*(x1 // 128)) + (x0 % 32)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')
