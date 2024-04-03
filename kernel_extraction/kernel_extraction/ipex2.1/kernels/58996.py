

# Original file: ./AllenaiLongformerBase__22_backward_143.5/AllenaiLongformerBase__22_backward_143.5.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/is/cisjlszmq6igi4ip224ycmyxzqegeuriq666mwg572pvtctp3c23.py
# Source Nodes: [], Original ATen: [aten.clone, aten.select_backward, aten.slice_backward]

triton_poi_fused_clone_select_backward_slice_backward_20 = async_compile.triton('triton_poi_fused_clone_select_backward_slice_backward_20', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_select_backward_slice_backward_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_select_backward_slice_backward_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 37822464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 513) % 512
    x0 = xindex % 513
    x2 = (xindex // 262656) % 3
    x3 = (xindex // 787968)
    x5 = (xindex // 513) % 1536
    tmp0 = x1
    tmp1 = tl.full([1], 256, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 257, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = x2
    tmp8 = tl.full([1], 3, tl.int32)
    tmp9 = tmp7 == tmp8
    tmp10 = 256 + x0
    tmp11 = tmp10 >= tmp1
    tmp12 = tmp11 & tmp6
    tmp13 = 0.0
    tmp14 = tl.where(tmp12, tmp13, 0.0)
    tmp15 = tl.full([1], 3, tl.int64)
    tmp16 = tl.full([1], 1, tl.int64)
    tmp17 = tmp15 >= tmp16
    tmp18 = tmp17 & tmp6
    tmp19 = tmp10 < tmp1
    tmp20 = tmp19 & tmp18
    tmp21 = tl.where(tmp20, tmp13, 0.0)
    tmp22 = tl.load(in_ptr0 + (18911488 + x0 + (513*x3) + (24624*x1)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.where(tmp19, tmp21, tmp22)
    tmp24 = tl.where(tmp18, tmp23, 0.0)
    tmp25 = tl.load(in_ptr0 + (18911488 + x0 + (513*x3) + (24624*x1)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.where(tmp17, tmp24, tmp25)
    tmp27 = tl.where(tmp11, tmp14, tmp26)
    tmp28 = tmp7 >= tmp16
    tmp29 = tmp28 & tmp6
    tmp30 = tmp19 & tmp29
    tmp31 = tl.where(tmp30, tmp13, 0.0)
    tmp32 = tl.load(in_ptr0 + (256 + x0 + (513*x3) + (24624*x1) + (6303744*x2)), tmp29, other=0.0)
    tmp33 = tl.where(tmp19, tmp31, tmp32)
    tmp34 = tl.where(tmp29, tmp33, 0.0)
    tmp35 = tl.load(in_ptr0 + (256 + x0 + (513*x3) + (24624*x1) + (6303744*x2)), tmp6, other=0.0)
    tmp36 = tl.where(tmp28, tmp34, tmp35)
    tmp37 = tl.where(tmp9, tmp27, tmp36)
    tmp38 = tl.where(tmp6, tmp37, 0.0)
    tmp39 = tl.where(tmp5, tmp38, tmp13)
    tmp40 = tl.where(tmp2, tmp39, 0.0)
    tmp41 = tl.where(tmp2, tmp40, tmp13)
    tl.store(out_ptr0 + (x0 + (513*x3) + (24624*x5)), tmp41, None)
''')
