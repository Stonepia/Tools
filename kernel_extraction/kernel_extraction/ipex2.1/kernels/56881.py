

# Original file: ./twins_pcpvt_base___60.0/twins_pcpvt_base___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/5b/c5bn6ssawvkxjz37lqmc6fnjk3jfdnxsc3ovmq4mkrtt3b7dcgt7.py
# Source Nodes: [scaled_dot_product_attention_3], Original ATen: [aten._scaled_dot_product_efficient_attention]
# scaled_dot_product_attention_3 => _scaled_dot_product_efficient_attention_3
triton_poi_fused__scaled_dot_product_efficient_attention_15 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_15', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_15(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp0, None)
''')
