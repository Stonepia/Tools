

# Original file: ./vision_maskrcnn__25_inference_65.5/vision_maskrcnn__25_inference_65.5.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/ff/cff5ihbqss5kkhmedzkg7idzn7dpd5hbqw4ao5htbewc73vd3wjo.py
# Source Nodes: [cat_9], Original ATen: [aten.cat]
# cat_9 => cat_5
triton_poi_fused_cat_26 = async_compile.triton('triton_poi_fused_cat_26', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i32', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_26', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_26(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 182400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = (xindex // 4)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (4*(x1 // 3))), xmask)
    tmp2 = tl.load(in_ptr1 + (x0 + (4*(x1 % 3))), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''')
