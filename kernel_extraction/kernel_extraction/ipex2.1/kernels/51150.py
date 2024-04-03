

# Original file: ./hf_Longformer__22_inference_62.2/hf_Longformer__22_inference_62.2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/tl/ctlyrbcb3f2upjwyg6q7rxk7qhm3fmloiueygqckp7sddy44fxc2.py
# Source Nodes: [new_zeros, setitem, setitem_1], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice, aten.slice_scatter]
# new_zeros => full
# setitem => copy, slice_5, slice_scatter, slice_scatter_1, slice_scatter_2, slice_scatter_3
# setitem_1 => copy_1, select_scatter, slice_22, slice_scatter_4, slice_scatter_5
triton_poi_fused_copy_new_zeros_select_scatter_slice_slice_scatter_4 = async_compile.triton('triton_poi_fused_copy_new_zeros_select_scatter_slice_slice_scatter_4', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_new_zeros_select_scatter_slice_slice_scatter_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_new_zeros_select_scatter_slice_slice_scatter_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25214976
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 131328) % 16
    x0 = xindex % 513
    x1 = (xindex // 513) % 256
    x3 = (xindex // 2101248)
    x5 = xindex % 131328
    x6 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 15, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 256, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = ((3808256 + x0 + (513*x1)) // 512) % 513
    tmp7 = tl.full([1], 512, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr0 + ((512*(((3808256 + x5) // 512) % 513)) + (262144*((3808256 + x5) // 262656)) + (3932160*x3) + (3932160*((3808256 + x5) // 3939840)) + (x5 % 512)), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tl.where(tmp9, tmp10, 0.0)
    tmp12 = tl.where(tmp5, tmp11, 0.0)
    tmp13 = tl.full([1], 15, tl.int64)
    tmp14 = tmp13 < tmp13
    tmp15 = tmp5 & tmp14
    tmp16 = ((3939584 + x0 + (513*x1)) // 512) % 513
    tmp17 = tmp16 < tmp7
    tmp18 = tmp17 & tmp15
    tmp19 = tl.load(in_ptr0 + ((262144*(((3939584 + x0 + (513*x1)) // 262656) % 15)) + (3932160*(((3939584 + x0 + (513*x1) + (3939840*x3)) // 3939840) % 12)) + ((3939584 + x0 + (513*x1)) % 262656)), tmp18, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = tl.where(tmp18, tmp19, 0.0)
    tmp21 = tl.where(tmp15, tmp20, 0.0)
    tmp22 = 0.0
    tmp23 = tl.where(tmp5, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp23, 0.0)
    tmp25 = tl.where(tmp14, tmp24, tmp22)
    tmp26 = tl.where(tmp5, tmp12, tmp25)
    tmp27 = tmp0 < tmp13
    tmp28 = tmp5 & tmp27
    tmp29 = (((-256) + x0 + (513*x1) + (262656*x2) + (3939840*x3)) // 512) % 513
    tmp30 = tmp29 < tmp7
    tmp31 = tmp30 & tmp28
    tmp32 = tl.load(in_ptr0 + ((262144*((((-256) + x0 + (513*x1) + (262656*x2) + (3939840*x3)) // 262656) % 180)) + (((-256) + x0 + (513*x1) + (262656*x2) + (3939840*x3)) % 262656)), tmp31, other=0.0).to(tl.float32)
    tmp33 = tl.where(tmp31, tmp32, 0.0)
    tmp34 = tl.where(tmp28, tmp33, 0.0)
    tmp35 = tl.where(tmp5, tmp34, tmp22)
    tmp36 = tl.where(tmp27, tmp35, 0.0)
    tmp37 = tl.where(tmp27, tmp36, tmp22)
    tmp38 = tl.where(tmp2, tmp26, tmp37)
    tl.store(out_ptr0 + (x6), tmp38, None)
''')
