

# Original file: ./sam___60.0/sam___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/an/canalb4b5vbzswbigsrjcatkkh5t7fqbueflo4ltydqcsiz5wetx.py
# Source Nodes: [interpolate], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.clamp, aten.mul, aten.rsub, aten.sub]
# interpolate => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_402, add_404, add_406, add_407, add_408, clamp_min, convert_element_type_736, convert_element_type_737, convert_element_type_739, iota_128, mul_413, mul_415, mul_417, mul_418, mul_419, mul_420, mul_421, mul_422, sub_186, sub_188, sub_189, sub_190, sub_191
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_65 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_65', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_65', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_65(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp22 = tl.load(in_ptr0 + (tmp21 + (256*tmp12)), None).to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp12.to(tl.float32)
    tmp25 = tmp11 - tmp24
    tmp26 = tmp2 - tmp25
    tmp27 = tmp23 * tmp26
    tmp28 = libdevice.ceil(tmp11)
    tmp29 = 255.0
    tmp30 = triton_helpers.minimum(tmp28, tmp29)
    tmp31 = tmp30.to(tl.int32)
    tmp32 = tl.load(in_ptr0 + (tmp21 + (256*tmp31)), None).to(tl.float32)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp33 * tmp25
    tmp35 = tmp27 + tmp34
    tmp36 = libdevice.ceil(tmp20)
    tmp37 = triton_helpers.minimum(tmp36, tmp29)
    tmp38 = tmp37.to(tl.int32)
    tmp39 = tl.load(in_ptr0 + (tmp38 + (256*tmp12)), None).to(tl.float32)
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp40 * tmp26
    tmp42 = tl.load(in_ptr0 + (tmp38 + (256*tmp31)), None).to(tl.float32)
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp43 * tmp25
    tmp45 = tmp41 + tmp44
    tmp46 = tmp21.to(tl.float32)
    tmp47 = tmp20 - tmp46
    tmp48 = tmp2 - tmp47
    tmp49 = tmp35 * tmp48
    tmp50 = tmp45 * tmp47
    tmp51 = tmp49 + tmp50
    tl.store(in_out_ptr0 + (x2), tmp51, None)
''')
