

# Original file: ./doctr_det_predictor___60.0/doctr_det_predictor___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/ox/coxoafmnfhf4b4lgmyejhmhewcoy4grn7shpfnmgthk2yfrjuzt5.py
# Source Nodes: [l__self___fpn_out_branches_0_3], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.mul, aten.sub]
# l__self___fpn_out_branches_0_3 => _unsafe_index_12, _unsafe_index_13, _unsafe_index_14, _unsafe_index_15, add_152, add_153, add_154, convert_element_type_257, convert_element_type_264, full_default, full_default_2, mul_209, mul_211, mul_213
triton_poi_fused__to_copy__unsafe_index_add_mul_sub_5 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_mul_sub_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_mul_sub_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_mul_sub_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp17 = tl.load(in_ptr0 + (x2 + (64*tmp16) + (16384*tmp7)), None).to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = libdevice.ceil(tmp6)
    tmp20 = triton_helpers.minimum(tmp19, tmp14)
    tmp21 = tmp20.to(tl.int32)
    tmp22 = tl.load(in_ptr0 + (x2 + (64*tmp16) + (16384*tmp21)), None).to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23 * tmp4
    tmp25 = tmp18 + tmp24
    tmp26 = tmp12.to(tl.int32)
    tmp27 = tl.load(in_ptr0 + (x2 + (64*tmp26) + (16384*tmp7)), None).to(tl.float32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.load(in_ptr0 + (x2 + (64*tmp26) + (16384*tmp21)), None).to(tl.float32)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp4
    tmp32 = tmp28 + tmp31
    tmp33 = tmp25 * tmp4
    tmp34 = tmp32 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp35, None)
''')
