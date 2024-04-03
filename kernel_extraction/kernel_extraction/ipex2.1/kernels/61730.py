

# Original file: ./densenet121___60.0/densenet121___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/5i/c5in472psxgcvzhw3npiqvhpbocufk5ppemuzvzch27abbz5tpng.py
# Source Nodes: [cat_65, cat_66, cat_67, cat_68, cat_69], Original ATen: [aten.cat]
# cat_65 => cat_54
# cat_66 => cat_53
# cat_67 => cat_52
# cat_68 => cat_51
# cat_69 => cat_50
triton_poi_fused_cat_106 = async_compile.triton('triton_poi_fused_cat_106', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_106', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_106(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + (800*x1)), tmp0, None)
    tl.store(out_ptr1 + (x0 + (832*x1)), tmp0, None)
    tl.store(out_ptr2 + (x0 + (864*x1)), tmp0, None)
    tl.store(out_ptr3 + (x0 + (896*x1)), tmp0, None)
    tl.store(out_ptr4 + (x0 + (928*x1)), tmp0, None)
''')
