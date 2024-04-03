

# Original file: ./AllenaiLongformerBase__22_backward_143.5/AllenaiLongformerBase__22_backward_143.5.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/v7/cv7xfpqcfppirifsiub6fcd6x6ebpz6xwbeksnk7v33fqoq46we2.py
# Source Nodes: [], Original ATen: [aten.add, aten.clone, aten.select_backward, aten.slice_backward]

triton_poi_fused_add_clone_select_backward_slice_backward_19 = async_compile.triton('triton_poi_fused_add_clone_select_backward_slice_backward_19', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_select_backward_slice_backward_19', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_clone_select_backward_slice_backward_19(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 37822464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 262656) % 3
    x1 = (xindex // 513) % 512
    x0 = xindex % 513
    x3 = (xindex // 787968)
    x5 = (xindex // 513) % 1536
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x1
    tmp4 = tl.full([1], 255, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (513*x3) + (24624*x1)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = 0.0
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tmp10 = tl.where(tmp2, tmp9, tmp8)
    tmp11 = tmp3 >= tmp4
    tmp12 = tl.full([1], 511, tl.int64)
    tmp13 = tmp3 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = x0
    tmp16 = tl.full([1], 257, tl.int64)
    tmp17 = tmp15 >= tmp16
    tmp18 = tmp17 & tmp14
    tmp19 = tl.load(in_ptr1 + (24367 + x0 + (513*x3) + (24624*x1) + (6303744*x2)), tmp18, other=0.0)
    tmp20 = tl.where(tmp18, tmp19, 0.0)
    tmp21 = tl.where(tmp17, tmp20, tmp8)
    tmp22 = tl.where(tmp14, tmp21, 0.0)
    tmp23 = tl.where(tmp14, tmp22, tmp8)
    tmp24 = tmp10 + tmp23
    tmp25 = tl.full([1], 2, tl.int32)
    tmp26 = tmp0 == tmp25
    tmp27 = tl.full([1], 256, tl.int64)
    tmp28 = tmp3 >= tmp27
    tmp29 = tmp15 < tmp16
    tmp30 = tmp29 & tmp28
    tmp31 = tl.full([1], 3, tl.int64)
    tmp32 = tl.full([1], 1, tl.int64)
    tmp33 = tmp31 >= tmp32
    tmp34 = tmp33 & tmp30
    tmp35 = 256 + x0
    tmp36 = tmp35 < tmp27
    tmp37 = tmp36 & tmp34
    tmp38 = tl.where(tmp37, tmp8, 0.0)
    tmp39 = tl.load(in_ptr1 + (12607744 + x0 + (513*x3) + (24624*x1)), tmp34, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.where(tmp36, tmp38, tmp39)
    tmp41 = tl.where(tmp34, tmp40, 0.0)
    tmp42 = tl.load(in_ptr1 + (12607744 + x0 + (513*x3) + (24624*x1)), tmp30, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.where(tmp33, tmp41, tmp42)
    tmp44 = tl.where(tmp30, tmp43, 0.0)
    tmp45 = tl.where(tmp29, tmp44, tmp8)
    tmp46 = tl.where(tmp28, tmp45, 0.0)
    tmp47 = tl.where(tmp28, tmp46, tmp8)
    tmp48 = tl.where(tmp26, tmp47, tmp8)
    tmp49 = tmp24 + tmp48
    tl.store(out_ptr0 + (x0 + (513*x3) + (24624*x5)), tmp49, None)
''')
