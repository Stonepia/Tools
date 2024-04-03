

# Original file: ./densenet121___60.0/densenet121___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/rf/crfgifgwzjqm4od3chqbbl4rigd72zykmbefqbi42rstbknf6gn4.py
# Source Nodes: [cat_104, cat_105, cat_106, cat_107, cat_108], Original ATen: [aten.cat]
# cat_104 => cat_17
# cat_105 => cat_16
# cat_106 => cat_15
# cat_107 => cat_14
# cat_108 => cat_13
triton_poi_fused_cat_30 = async_compile.triton('triton_poi_fused_cat_30', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_30', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_30(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (384*x1)), tmp0, None)
    tl.store(out_ptr1 + (x0 + (416*x1)), tmp0, None)
    tl.store(out_ptr2 + (x0 + (448*x1)), tmp0, None)
    tl.store(out_ptr3 + (x0 + (480*x1)), tmp0, None)
    tl.store(out_ptr4 + (x0 + (512*x1)), tmp0, None)
''')
