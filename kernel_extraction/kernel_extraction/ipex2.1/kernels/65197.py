

# Original file: ./pytorch_unet___60.0/pytorch_unet___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/ny/cny3votgsyqjgfp4hxsxvhgrxj64uhiylmbqov74m5jkikwlboay.py
# Source Nodes: [l__mod___up3_up], Original ATen: [aten._unsafe_index, aten.add, aten.mul, aten.rsub, aten.sub]
# l__mod___up3_up => _unsafe_index_10, _unsafe_index_11, add_41, mul_68, mul_69, sub_22, sub_23
triton_poi_fused__unsafe_index_add_mul_rsub_sub_14 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_rsub_sub_14', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_rsub_sub_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_rsub_sub_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19578880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 478) % 320
    x0 = xindex % 478
    x2 = (xindex // 152960)
    x4 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.49843260188087773
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.int32)
    tmp9 = x0
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp11 + tmp4
    tmp13 = 0.4989517819706499
    tmp14 = tmp12 * tmp13
    tmp15 = libdevice.ceil(tmp14)
    tmp16 = 238.0
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tmp18 = tmp17.to(tl.int32)
    tmp19 = tl.load(in_ptr0 + (x2 + (128*tmp18) + (30592*tmp8)), None)
    tmp20 = tmp8.to(tl.float32)
    tmp21 = tmp7 - tmp20
    tmp22 = tmp2 - tmp21
    tmp23 = tmp19 * tmp22
    tmp24 = libdevice.ceil(tmp7)
    tmp25 = 159.0
    tmp26 = triton_helpers.minimum(tmp24, tmp25)
    tmp27 = tmp26.to(tl.int32)
    tmp28 = tl.load(in_ptr0 + (x2 + (128*tmp18) + (30592*tmp27)), None)
    tmp29 = tmp28 * tmp21
    tmp30 = tmp23 + tmp29
    tl.store(out_ptr0 + (x4), tmp30, None)
''')
