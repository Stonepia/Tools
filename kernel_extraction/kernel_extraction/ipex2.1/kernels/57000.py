

# Original file: ./twins_pcpvt_base___60.0/twins_pcpvt_base___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/26/c265zdmzgqml7wsksik2p4klreqzclbicmnizdxw3nq6jxnchwze.py
# Source Nodes: [scaled_dot_product_attention_7], Original ATen: [aten._scaled_dot_product_efficient_attention]
# scaled_dot_product_attention_7 => _scaled_dot_product_efficient_attention_7
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

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_29', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_29(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 320
    x1 = (xindex // 320)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (320 + x0 + (640*x1)), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')
