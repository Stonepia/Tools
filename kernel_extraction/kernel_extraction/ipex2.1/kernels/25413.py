

# Original file: ./sam___60.0/sam___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/hp/chp5ehqd55b4txaxhahvvgmt77waywavuzgupzg26sergw664m7c.py
# Source Nodes: [interpolate], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.clamp, aten.mul, aten.rsub, aten.sub]
# interpolate => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_402, add_404, add_406, add_407, add_408, clamp_min, convert_element_type_64, convert_element_type_66, iota_128, mul_413, mul_415, mul_417, mul_418, mul_419, mul_420, mul_421, mul_422, sub_186, sub_188, sub_189, sub_190, sub_191
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_67 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_67', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_67', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_67(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 1024)
    x0 = xindex % 1024
    x2 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.5
    tmp7 = tmp5 + tmp6
    tmp8 = 0.25
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9 - tmp6
    tmp11 = triton_helpers.maximum(tmp10, tmp4)
    tmp12 = tmp11.to(tl.int32)
    tmp13 = x0
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 * tmp2
    tmp16 = tmp15 + tmp4
    tmp17 = tmp16 + tmp6
    tmp18 = tmp17 * tmp8
    tmp19 = tmp18 - tmp6
    tmp20 = triton_helpers.maximum(tmp19, tmp4)
    tmp21 = tmp20.to(tl.int32)
    tmp22 = tl.load(in_ptr0 + (tmp21 + (256*tmp12)), None)
    tmp23 = tmp12.to(tl.float32)
    tmp24 = tmp11 - tmp23
    tmp25 = tmp2 - tmp24
    tmp26 = tmp22 * tmp25
    tmp27 = libdevice.ceil(tmp11)
    tmp28 = 255.0
    tmp29 = triton_helpers.minimum(tmp27, tmp28)
    tmp30 = tmp29.to(tl.int32)
    tmp31 = tl.load(in_ptr0 + (tmp21 + (256*tmp30)), None)
    tmp32 = tmp31 * tmp24
    tmp33 = tmp26 + tmp32
    tmp34 = libdevice.ceil(tmp20)
    tmp35 = triton_helpers.minimum(tmp34, tmp28)
    tmp36 = tmp35.to(tl.int32)
    tmp37 = tl.load(in_ptr0 + (tmp36 + (256*tmp12)), None)
    tmp38 = tmp37 * tmp25
    tmp39 = tl.load(in_ptr0 + (tmp36 + (256*tmp30)), None)
    tmp40 = tmp39 * tmp24
    tmp41 = tmp38 + tmp40
    tmp42 = tmp21.to(tl.float32)
    tmp43 = tmp20 - tmp42
    tmp44 = tmp2 - tmp43
    tmp45 = tmp33 * tmp44
    tmp46 = tmp41 * tmp43
    tmp47 = tmp45 + tmp46
    tl.store(in_out_ptr0 + (x2), tmp47, None)
''')
