

# Original file: ./densenet121___60.0/densenet121___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/j3/cj322plael7lzhvwtlev6yqejxtyr3wz5mi7362jqosvjgjhasft.py
# Source Nodes: [cat_70, cat_71, cat_72, cat_73, cat_74, cat_75, cat_76], Original ATen: [aten.cat]
# cat_70 => cat_49
# cat_71 => cat_48
# cat_72 => cat_47
# cat_73 => cat_46
# cat_74 => cat_45
# cat_75 => cat_44
# cat_76 => cat_43
triton_poi_fused_cat_92 = async_compile.triton('triton_poi_fused_cat_92', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_92', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_92(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + (576*x1)), tmp0, None)
    tl.store(out_ptr1 + (x0 + (608*x1)), tmp0, None)
    tl.store(out_ptr2 + (x0 + (640*x1)), tmp0, None)
    tl.store(out_ptr3 + (x0 + (672*x1)), tmp0, None)
    tl.store(out_ptr4 + (x0 + (704*x1)), tmp0, None)
    tl.store(out_ptr5 + (x0 + (736*x1)), tmp0, None)
    tl.store(out_ptr6 + (x0 + (768*x1)), tmp0, None)
''')