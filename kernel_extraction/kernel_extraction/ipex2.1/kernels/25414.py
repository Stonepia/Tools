

# Original file: ./sam___60.0/sam___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/42/c42ywe63fkqwz77fitbvb3cr4hgr42irz3vlxz5yplsf5vjbulcj.py
# Source Nodes: [gt, interpolate_1], Original ATen: [aten._unsafe_index, aten.add, aten.gt, aten.mul, aten.sub]
# gt => gt
# interpolate_1 => _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, add_413, add_414, add_415, full_default_1, full_default_3, mul_428, mul_430, mul_432
triton_poi_fused__unsafe_index_add_gt_mul_sub_68 = async_compile.triton('triton_poi_fused__unsafe_index_add_gt_mul_sub_68', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i1', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_gt_mul_sub_68', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__unsafe_index_add_gt_mul_sub_68(in_ptr0, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256)
    x0 = xindex % 256
    x2 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.5
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp2
    tmp9 = tmp8 - tmp6
    tmp10 = triton_helpers.maximum(tmp9, tmp4)
    tmp11 = tmp10.to(tl.int32)
    tmp12 = x0
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp2
    tmp15 = tmp14 + tmp4
    tmp16 = tmp15 + tmp6
    tmp17 = tmp16 * tmp2
    tmp18 = tmp17 - tmp6
    tmp19 = triton_helpers.maximum(tmp18, tmp4)
    tmp20 = tmp19.to(tl.int32)
    tmp21 = tl.load(in_ptr0 + (tmp20 + (1024*tmp11)), None)
    tmp22 = libdevice.ceil(tmp10)
    tmp23 = 255.0
    tmp24 = triton_helpers.minimum(tmp22, tmp23)
    tmp25 = tmp24.to(tl.int32)
    tmp26 = tl.load(in_ptr0 + (tmp20 + (1024*tmp25)), None)
    tmp27 = tmp26 * tmp4
    tmp28 = tmp21 + tmp27
    tmp29 = libdevice.ceil(tmp19)
    tmp30 = triton_helpers.minimum(tmp29, tmp23)
    tmp31 = tmp30.to(tl.int32)
    tmp32 = tl.load(in_ptr0 + (tmp31 + (1024*tmp11)), None)
    tmp33 = tl.load(in_ptr0 + (tmp31 + (1024*tmp25)), None)
    tmp34 = tmp33 * tmp4
    tmp35 = tmp32 + tmp34
    tmp36 = tmp35 * tmp4
    tmp37 = tmp28 + tmp36
    tmp38 = tmp37 > tmp4
    tl.store(out_ptr2 + (x2), tmp38, None)
''')
