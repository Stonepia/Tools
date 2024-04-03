

# Original file: ./pit_b_224___60.0/pit_b_224___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/3x/c3xwqje3uyss5ni76j4uu3v6qr32ul74pjcqlw3fzt2yubectbqu.py
# Source Nodes: [scaled_dot_product_attention_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
# scaled_dot_product_attention_9 => _scaled_dot_product_efficient_attention_9
triton_poi_fused__scaled_dot_product_efficient_attention_29 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_29', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_29', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_29(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4259840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3072*x1)), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')
