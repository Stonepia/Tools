

# Original file: ./xcit_large_24_p8_224___60.0/xcit_large_24_p8_224___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/em/cembmjxkkomxbgzoscb32l3lstgudat3ycjctolyp5rgfzl5o2qa.py
# Source Nodes: [cat_9, l__self___cls_attn_blocks_0_attn_proj_drop], Original ATen: [aten._to_copy, aten.clone]
# cat_9 => convert_element_type_685
# l__self___cls_attn_blocks_0_attn_proj_drop => clone_243
triton_poi_fused__to_copy_clone_31 = async_compile.triton('triton_poi_fused__to_copy_clone_31', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_31', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_clone_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    x1 = (xindex // 768)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + (602880*x1)), tmp1, xmask)
''')
