

# Original file: ./AllenaiLongformerBase__22_forward_137.2/AllenaiLongformerBase__22_forward_137.2_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/cx/ccxmeovt2puotuawv54gauxcw575eaasvkvfsylfayy4ti5ewmhb.py
# Source Nodes: [new_zeros_1, setitem_6, setitem_7, setitem_8, setitem_9], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice, aten.slice_scatter]
# new_zeros_1 => full_5
# setitem_6 => copy_6, slice_139, slice_scatter_22, slice_scatter_23, slice_scatter_24, slice_scatter_25
# setitem_7 => select_scatter_2, slice_156, slice_scatter_27, slice_scatter_28
# setitem_8 => slice_172, slice_scatter_30, slice_scatter_31, slice_scatter_32
# setitem_9 => select_scatter_3, slice_189, slice_scatter_34
triton_poi_fused_copy_new_zeros_select_scatter_slice_slice_scatter_18 = async_compile.triton('triton_poi_fused_copy_new_zeros_select_scatter_slice_slice_scatter_18', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_new_zeros_select_scatter_slice_slice_scatter_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_new_zeros_select_scatter_slice_slice_scatter_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2101248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 131328) % 4
    x3 = (xindex // 525312)
    x4 = xindex % 131328
    x5 = xindex % 525312
    x0 = xindex % 513
    x1 = (xindex // 513) % 256
    x6 = xindex
    tmp3 = tl.load(in_ptr0 + (x4 + (131328*x3)), None, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (x4 + (131328*x3)), None, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp0 >= tmp4
    tmp6 = tl.load(in_ptr1 + ((-131328) + x5 + (393984*x3)), tmp5, other=0.0).to(tl.float32)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = tl.full([1], 3, tl.int32)
    tmp9 = tmp0 == tmp8
    tmp11 = tl.full([1], 3, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = x0
    tmp14 = tl.full([1], 256, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tmp15 & tmp12
    tmp17 = (((-256) + x0 + (513*x1) + (262656*x2) + (787968*x3)) // 512) % 513
    tmp18 = tl.full([1], 512, tl.int64)
    tmp19 = tmp17 < tmp18
    tmp20 = tmp19 & tmp16
    tmp21 = tl.load(in_ptr3 + ((256*((((-256) + x0 + (513*x1) + (262656*x2) + (787968*x3)) // 262656) % 3)) + (1024*((((-256) + x0 + (513*x1) + (262656*x2) + (787968*x3)) // 787968) % 4)) + ((((-256) + x0 + (513*x1) + (262656*x2) + (787968*x3)) // 512) % 513)), tmp20, other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr4 + ((256*((((-256) + x0 + (513*x1) + (262656*x2) + (787968*x3)) // 262656) % 3)) + (1024*((((-256) + x0 + (513*x1) + (262656*x2) + (787968*x3)) // 787968) % 4)) + (((-256) + x0 + (513*x1) + (262656*x2) + (787968*x3)) % 512)), tmp20, other=0.0).to(tl.float32)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.where(tmp20, tmp23, 0.0)
    tmp25 = tl.where(tmp16, tmp24, 0.0)
    tmp26 = 0.0
    tmp27 = tl.where(tmp15, tmp25, tmp26)
    tmp28 = tl.where(tmp12, tmp27, 0.0)
    tmp29 = tl.where(tmp12, tmp28, tmp26)
    tmp30 = tl.where(tmp9, tmp10, tmp29)
    tmp31 = tl.where(tmp5, tmp7, tmp30)
    tmp32 = tl.where(tmp2, tmp3, tmp31)
    tl.store(out_ptr0 + (x6), tmp32, None)
''')
