

# Original file: ./pytorch_unet___60.0/pytorch_unet___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/4a/c4aexyji2yigvmvbmo5qzfrtjcnaygzefb55vt4b3ovla7wrw3jc.py
# Source Nodes: [l__self___up4_up], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.mul, aten.rsub, aten.sub]
# l__self___up4_up => _unsafe_index_14, _unsafe_index_15, add_50, convert_element_type_105, mul_84, mul_85, sub_28, sub_29
triton_poi_fused__to_copy__unsafe_index_add_mul_rsub_sub_18 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_mul_rsub_sub_18', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_mul_rsub_sub_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_mul_rsub_sub_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 39239680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 958) % 640
    x0 = xindex % 958
    x2 = (xindex // 613120)
    x4 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.49921752738654146
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.int32)
    tmp9 = x0
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp11 + tmp4
    tmp13 = 0.4994775339602926
    tmp14 = tmp12 * tmp13
    tmp15 = libdevice.ceil(tmp14)
    tmp16 = 478.0
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tmp18 = tmp17.to(tl.int32)
    tmp19 = tl.load(in_ptr0 + (x2 + (64*tmp18) + (30656*tmp8)), None).to(tl.float32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp8.to(tl.float32)
    tmp22 = tmp7 - tmp21
    tmp23 = tmp2 - tmp22
    tmp24 = tmp20 * tmp23
    tmp25 = libdevice.ceil(tmp7)
    tmp26 = 319.0
    tmp27 = triton_helpers.minimum(tmp25, tmp26)
    tmp28 = tmp27.to(tl.int32)
    tmp29 = tl.load(in_ptr0 + (x2 + (64*tmp18) + (30656*tmp28)), None).to(tl.float32)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp22
    tmp32 = tmp24 + tmp31
    tl.store(out_ptr0 + (x4), tmp32, None)
''')
