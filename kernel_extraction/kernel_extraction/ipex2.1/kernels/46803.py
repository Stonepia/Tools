

# Original file: ./DALLE2_pytorch__21_inference_61.1/DALLE2_pytorch__21_inference_61.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/jr/cjrybwtb7uzoavzfzzo3vuspf6jjt2pmbtqbsnqn6sz3uhqk45gw.py
# Source Nodes: [and_, clone, pad, rearrange, where], Original ATen: [aten.bitwise_and, aten.clone, aten.constant_pad_nd, aten.view, aten.where]
# and_ => bitwise_and
# clone => clone
# pad => constant_pad_nd
# rearrange => full_default
# where => where
triton_poi_fused_bitwise_and_clone_constant_pad_nd_view_where_2 = async_compile.triton('triton_poi_fused_bitwise_and_clone_constant_pad_nd_view_where_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_clone_constant_pad_nd_view_where_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_bitwise_and_clone_constant_pad_nd_view_where_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 512) % 256
    x2 = (xindex // 131072)
    x3 = xindex % 131072
    tmp9 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 77, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x1 + (77*x2)), tmp2, eviction_policy='evict_last')
    tmp4 = tl.where(tmp2, tmp3, False)
    tmp5 = tl.full([1], True, tl.int1)
    tmp6 = tmp4 & tmp5
    tmp7 = tl.load(in_ptr1 + (x3 + (39424*x2)), tmp2, other=0.0)
    tmp8 = tl.where(tmp2, tmp7, 0.0)
    tmp10 = tl.where(tmp6, tmp8, tmp9)
    tl.store(out_ptr0 + (x3 + (133120*x2)), tmp10, None)
''')