

# Original file: ./detectron2_fcos_r_50_fpn__75_inference_115.55/detectron2_fcos_r_50_fpn__75_inference_115.55_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/46/c465e72hdxuc7o3xjdd5on732rc2ss5zc5epgw4g24luq3yky7zf.py
# Source Nodes: [add_2, add_3, setitem, setitem_1, setitem_2, setitem_3, sub_2, sub_3, to_1, zeros_like], Original ATen: [aten._to_copy, aten.add, aten.copy, aten.slice, aten.slice_scatter, aten.sub, aten.zeros_like]
# add_2 => add_2
# add_3 => add_3
# setitem => copy, slice_18, slice_scatter, slice_scatter_1
# setitem_1 => copy_1, slice_scatter_2, slice_scatter_3
# setitem_2 => copy_2, slice_scatter_4, slice_scatter_5
# setitem_3 => copy_3, slice_scatter_6, slice_scatter_7
# sub_2 => sub_2
# sub_3 => sub_3
# to_1 => convert_element_type_1
# zeros_like => full
triton_poi_fused__to_copy_add_copy_slice_slice_scatter_sub_zeros_like_1 = async_compile.triton('triton_poi_fused__to_copy_add_copy_slice_slice_scatter_sub_zeros_like_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i64', 2: '*i64', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp32', 7: 'i32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_copy_slice_slice_scatter_sub_zeros_like_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 9), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_copy_slice_slice_scatter_sub_zeros_like_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = (xindex // 4)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.load(in_ptr0 + (x1), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x1), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.where(tmp4 < 0, tmp4 + ks0, tmp4)
    # tl.device_assert(((0 <= tmp5) & (tmp5 < ks0)) | ~(xmask & tmp2), "index out of bounds: 0 <= tmp5 < ks0")
    tmp6 = tl.load(in_ptr2 + (tl.broadcast_to(2*tmp5, [XBLOCK])), tmp2 & xmask, other=0.0)
    tmp7 = tl.where(tmp6 < 0, tmp6 + ks1, tmp6)
    # tl.device_assert(((0 <= tmp7) & (tmp7 < ks1)) | ~(xmask & tmp2), "index out of bounds: 0 <= tmp7 < ks1")
    tmp8 = tl.load(in_ptr3 + (tl.broadcast_to(4*tmp7, [XBLOCK])), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp9 = triton_helpers.maximum(0, tmp8)
    tmp10 = tl.load(in_ptr4 + (4*x1), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 - tmp11
    tmp13 = tl.where(tmp2, tmp12, 0.0)
    tmp14 = 0.0
    tmp15 = tl.where(tmp2, tmp13, tmp14)
    tmp16 = tl.full([1], 1, tl.int64)
    tmp17 = tmp0 >= tmp16
    tmp18 = ((-1) + x0) % 4
    tmp19 = tmp18 == tmp1
    tmp20 = tmp17 & tmp19
    tmp21 = tl.load(in_ptr5 + (x1), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr1 + (x1), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.where(tmp22 < 0, tmp22 + ks0, tmp22)
    # tl.device_assert(((0 <= tmp23) & (tmp23 < ks0)) | ~(tmp20 & xmask), "index out of bounds: 0 <= tmp23 < ks0")
    tmp24 = tl.load(in_ptr2 + (tl.broadcast_to(2*tmp23, [XBLOCK])), tmp20 & xmask, other=0.0)
    tmp25 = tl.where(tmp24 < 0, tmp24 + ks1, tmp24)
    # tl.device_assert(((0 <= tmp25) & (tmp25 < ks1)) | ~(tmp20 & xmask), "index out of bounds: 0 <= tmp25 < ks1")
    tmp26 = tl.load(in_ptr3 + (tl.broadcast_to(1 + (4*tmp25), [XBLOCK])), tmp20 & xmask, other=0.0).to(tl.float32)
    tmp27 = triton_helpers.maximum(0, tmp26)
    tmp28 = tl.load(in_ptr4 + (1 + (4*x1)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp21 - tmp29
    tmp31 = tl.where(tmp20, tmp30, 0.0)
    tmp32 = tl.where(tmp20, tmp31, tmp15)
    tmp33 = tl.full([1], 2, tl.int64)
    tmp34 = tmp0 >= tmp33
    tmp35 = ((-2) + x0) % 4
    tmp36 = tmp35 == tmp1
    tmp37 = tmp34 & tmp36
    tmp38 = tl.load(in_ptr0 + (x1), tmp37 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr1 + (x1), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.where(tmp39 < 0, tmp39 + ks0, tmp39)
    # tl.device_assert(((0 <= tmp40) & (tmp40 < ks0)) | ~(xmask & tmp37), "index out of bounds: 0 <= tmp40 < ks0")
    tmp41 = tl.load(in_ptr2 + (tl.broadcast_to(2*tmp40, [XBLOCK])), tmp37 & xmask, other=0.0)
    tmp42 = tl.where(tmp41 < 0, tmp41 + ks1, tmp41)
    # tl.device_assert(((0 <= tmp42) & (tmp42 < ks1)) | ~(xmask & tmp37), "index out of bounds: 0 <= tmp42 < ks1")
    tmp43 = tl.load(in_ptr3 + (tl.broadcast_to(2 + (4*tmp42), [XBLOCK])), tmp37 & xmask, other=0.0).to(tl.float32)
    tmp44 = triton_helpers.maximum(0, tmp43)
    tmp45 = tl.load(in_ptr4 + (2 + (4*x1)), tmp37 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp46 = tmp44 * tmp45
    tmp47 = tmp38 + tmp46
    tmp48 = tl.where(tmp37, tmp47, 0.0)
    tmp49 = tl.where(tmp37, tmp48, tmp32)
    tmp50 = tl.full([1], 3, tl.int64)
    tmp51 = tmp0 >= tmp50
    tmp52 = ((-3) + x0) % 4
    tmp53 = tmp52 == tmp1
    tmp54 = tmp51 & tmp53
    tmp55 = tl.load(in_ptr5 + (x1), tmp54 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp56 = tl.load(in_ptr1 + (x1), tmp54 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.where(tmp56 < 0, tmp56 + ks0, tmp56)
    # tl.device_assert(((0 <= tmp57) & (tmp57 < ks0)) | ~(tmp54 & xmask), "index out of bounds: 0 <= tmp57 < ks0")
    tmp58 = tl.load(in_ptr2 + (tl.broadcast_to(2*tmp57, [XBLOCK])), tmp54 & xmask, other=0.0)
    tmp59 = tl.where(tmp58 < 0, tmp58 + ks1, tmp58)
    # tl.device_assert(((0 <= tmp59) & (tmp59 < ks1)) | ~(tmp54 & xmask), "index out of bounds: 0 <= tmp59 < ks1")
    tmp60 = tl.load(in_ptr3 + (tl.broadcast_to(3 + (4*tmp59), [XBLOCK])), tmp54 & xmask, other=0.0).to(tl.float32)
    tmp61 = triton_helpers.maximum(0, tmp60)
    tmp62 = tl.load(in_ptr4 + (3 + (4*x1)), tmp54 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp63 = tmp61 * tmp62
    tmp64 = tmp55 + tmp63
    tmp65 = tl.where(tmp54, tmp64, 0.0)
    tmp66 = tl.where(tmp54, tmp65, tmp49)
    tmp67 = tmp66.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp67, xmask)
''')
