

# Original file: ./dpn107___60.0/dpn107___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/cc/cccja25os3xucj6cf7ygxny6xkfkf3dmw4wntsx4kvcsblejdx7u.py
# Source Nodes: [cat_95, cat_96], Original ATen: [aten.cat]
# cat_95 => cat_44
# cat_96 => cat_43
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

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_92', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_92(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 150528
    x1 = (xindex // 150528)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (351232*x1)), tmp0, None)
    tl.store(out_ptr1 + (x0 + (163072*x1)), tmp0, None)
''')