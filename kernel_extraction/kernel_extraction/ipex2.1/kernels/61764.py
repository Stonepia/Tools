

# Original file: ./densenet121___60.0/densenet121___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/ll/cllnjuqlf3bbash5vkhl5oajbim6cu72hzzbaaoj6jxuwdypjbje.py
# Source Nodes: [cat_108, cat_109, cat_110, cat_111, cat_112, cat_113, cat_114], Original ATen: [aten.cat]
# cat_108 => cat_13
# cat_109 => cat_12
# cat_110 => cat_11
# cat_111 => cat_10
# cat_112 => cat_9
# cat_113 => cat_8
# cat_114 => cat_7
triton_poi_fused_cat_18 = async_compile.triton('triton_poi_fused_cat_18', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_18(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (192*x1)), tmp0, None)
    tl.store(out_ptr1 + (x0 + (224*x1)), tmp0, None)
    tl.store(out_ptr2 + (x0 + (256*x1)), tmp0, None)
    tl.store(out_ptr3 + (x0 + (288*x1)), tmp0, None)
    tl.store(out_ptr4 + (x0 + (320*x1)), tmp0, None)
    tl.store(out_ptr5 + (x0 + (352*x1)), tmp0, None)
    tl.store(out_ptr6 + (x0 + (384*x1)), tmp0, None)
''')
