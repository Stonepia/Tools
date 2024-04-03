

# Original file: ./inception_v3___60.0/inception_v3___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/tm/ctmatqz7s6kczuixelbzgbboyhlhjjn4utxkr5fkocd6cv3wczun.py
# Source Nodes: [max_pool2d], Original ATen: [aten.max_pool2d_with_indices]
# max_pool2d => max_pool2d_with_indices_2
triton_poi_fused_max_pool2d_with_indices_11 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_11', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10653696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 288
    x1 = (xindex // 288) % 17
    x2 = (xindex // 4896) % 17
    x3 = (xindex // 83232)
    x4 = (xindex // 288)
    tmp0 = tl.load(in_ptr0 + (x0 + (576*x1) + (20160*x2) + (352800*x3)), None)
    tmp1 = tl.load(in_ptr0 + (288 + x0 + (576*x1) + (20160*x2) + (352800*x3)), None)
    tmp3 = tl.load(in_ptr0 + (576 + x0 + (576*x1) + (20160*x2) + (352800*x3)), None)
    tmp5 = tl.load(in_ptr0 + (10080 + x0 + (576*x1) + (20160*x2) + (352800*x3)), None)
    tmp7 = tl.load(in_ptr0 + (10368 + x0 + (576*x1) + (20160*x2) + (352800*x3)), None)
    tmp9 = tl.load(in_ptr0 + (10656 + x0 + (576*x1) + (20160*x2) + (352800*x3)), None)
    tmp11 = tl.load(in_ptr0 + (20160 + x0 + (576*x1) + (20160*x2) + (352800*x3)), None)
    tmp13 = tl.load(in_ptr0 + (20448 + x0 + (576*x1) + (20160*x2) + (352800*x3)), None)
    tmp15 = tl.load(in_ptr0 + (20736 + x0 + (576*x1) + (20160*x2) + (352800*x3)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (x0 + (768*x4)), tmp16, None)
''')
