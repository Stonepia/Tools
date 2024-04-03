

# Original file: ./doctr_det_predictor___60.0/doctr_det_predictor___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/sv/csviyo3puuvtsetl4wwr2rlmecfgygawzgnyg4m3jsdpeefder3g.py
# Source Nodes: [l__self___fpn_out_branches_0_3], Original ATen: [aten._unsafe_index, aten.add, aten.mul, aten.sub]
# l__self___fpn_out_branches_0_3 => _unsafe_index_12, _unsafe_index_13, _unsafe_index_14, _unsafe_index_15, add_152, add_153, add_154, full_default, full_default_2, mul_209, mul_211, mul_213
triton_poi_fused__unsafe_index_add_mul_sub_5 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_5(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256) % 256
    x0 = xindex % 256
    x2 = (xindex // 65536)
    x4 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5 * tmp2
    tmp7 = tmp6.to(tl.int32)
    tmp8 = x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 * tmp2
    tmp11 = tmp10 + tmp4
    tmp12 = tmp11 * tmp2
    tmp13 = libdevice.ceil(tmp12)
    tmp14 = 255.0
    tmp15 = triton_helpers.minimum(tmp13, tmp14)
    tmp16 = tmp15.to(tl.int32)
    tmp17 = tl.load(in_ptr0 + (x2 + (64*tmp16) + (16384*tmp7)), None)
    tmp18 = libdevice.ceil(tmp6)
    tmp19 = triton_helpers.minimum(tmp18, tmp14)
    tmp20 = tmp19.to(tl.int32)
    tmp21 = tl.load(in_ptr0 + (x2 + (64*tmp16) + (16384*tmp20)), None)
    tmp22 = tmp21 * tmp4
    tmp23 = tmp17 + tmp22
    tmp24 = tmp12.to(tl.int32)
    tmp25 = tl.load(in_ptr0 + (x2 + (64*tmp24) + (16384*tmp7)), None)
    tmp26 = tl.load(in_ptr0 + (x2 + (64*tmp24) + (16384*tmp20)), None)
    tmp27 = tmp26 * tmp4
    tmp28 = tmp25 + tmp27
    tmp29 = tmp23 * tmp4
    tmp30 = tmp28 + tmp29
    tl.store(out_ptr1 + (x4), tmp30, None)
''')
