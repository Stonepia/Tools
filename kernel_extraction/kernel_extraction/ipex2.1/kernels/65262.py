

# Original file: ./pytorch_unet___60.0/pytorch_unet___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/nc/cncny3ewmk4lvkwtpjrcll5mpjwwgwxgz5aie2ueewipcr57ei2b.py
# Source Nodes: [l__self___up3_up], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.mul, aten.rsub, aten.sub]
# l__self___up3_up => _unsafe_index_8, _unsafe_index_9, add_40, convert_element_type_87, mul_66, mul_67, sub_22, sub_23
triton_poi_fused__to_copy__unsafe_index_add_mul_rsub_sub_13 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_mul_rsub_sub_13', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_mul_rsub_sub_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_mul_rsub_sub_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp15 = tmp14.to(tl.int32)
    tmp16 = tl.load(in_ptr0 + (x2 + (128*tmp15) + (30592*tmp8)), None).to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp8.to(tl.float32)
    tmp19 = tmp7 - tmp18
    tmp20 = tmp2 - tmp19
    tmp21 = tmp17 * tmp20
    tmp22 = libdevice.ceil(tmp7)
    tmp23 = 159.0
    tmp24 = triton_helpers.minimum(tmp22, tmp23)
    tmp25 = tmp24.to(tl.int32)
    tmp26 = tl.load(in_ptr0 + (x2 + (128*tmp15) + (30592*tmp25)), None).to(tl.float32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27 * tmp19
    tmp29 = tmp21 + tmp28
    tl.store(out_ptr0 + (x4), tmp29, None)
''')