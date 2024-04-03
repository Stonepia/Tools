

# Original file: ./densenet121___60.0/densenet121___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/ws/cwsomyefgb5ma7vxstdj24w6e6juk4ty2kyy4s3qbi3ktsqyam4r.py
# Source Nodes: [cat_117, cat_118, cat_119, cat_120, cat_121, cat_122], Original ATen: [aten.cat]
# cat_117 => cat_5
# cat_118 => cat_4
# cat_119 => cat_3
# cat_120 => cat_2
# cat_121 => cat_1
# cat_122 => cat
triton_poi_fused_cat_3 = async_compile.triton('triton_poi_fused_cat_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_3(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (96*x1)), tmp0, None)
    tl.store(out_ptr1 + (x0 + (128*x1)), tmp0, None)
    tl.store(out_ptr2 + (x0 + (160*x1)), tmp0, None)
    tl.store(out_ptr3 + (x0 + (192*x1)), tmp0, None)
    tl.store(out_ptr4 + (x0 + (224*x1)), tmp0, None)
    tl.store(out_ptr5 + (x0 + (256*x1)), tmp0, None)
''')
