

# Original file: ./hrnet_w18___60.0/hrnet_w18___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/ge/cge5ktf67hvuk6fuvauvrzc45ptxjrh6rwwbfnv5fltlorita5zd.py
# Source Nodes: [add_26, add_27, add_28, l__mod___stage4_0_fuse_layers_0_1_2, l__mod___stage4_0_fuse_layers_0_2_2, l__mod___stage4_0_fuse_layers_0_3_2], Original ATen: [aten._unsafe_index, aten.add]
# add_26 => add_518
# add_27 => add_523
# add_28 => add_528
# l__mod___stage4_0_fuse_layers_0_1_2 => _unsafe_index_13
# l__mod___stage4_0_fuse_layers_0_2_2 => _unsafe_index_14
# l__mod___stage4_0_fuse_layers_0_3_2 => _unsafe_index_15
triton_poi_fused__unsafe_index_add_5 = async_compile.triton('triton_poi_fused__unsafe_index_add_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused__unsafe_index_add_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7225344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x2 = (xindex // 1008) % 56
    x1 = (xindex // 18) % 56
    x0 = xindex % 18
    x3 = (xindex // 56448)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = x2
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.int32)
    tmp10 = x1
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 * tmp3
    tmp13 = tmp12 + tmp5
    tmp14 = tmp13 * tmp7
    tmp15 = tmp14.to(tl.int32)
    tmp16 = tl.load(in_ptr1 + (x0 + (18*tmp15) + (504*tmp9) + (14112*x3)), None)
    tmp17 = tmp0 + tmp16
    tmp18 = 0.25
    tmp19 = tmp6 * tmp18
    tmp20 = tmp19.to(tl.int32)
    tmp21 = tmp13 * tmp18
    tmp22 = tmp21.to(tl.int32)
    tmp23 = tl.load(in_ptr2 + (x0 + (18*tmp22) + (252*tmp20) + (3528*x3)), None)
    tmp24 = tmp17 + tmp23
    tmp25 = 0.125
    tmp26 = tmp6 * tmp25
    tmp27 = tmp26.to(tl.int32)
    tmp28 = tmp13 * tmp25
    tmp29 = tmp28.to(tl.int32)
    tmp30 = tl.load(in_ptr3 + (x0 + (18*tmp29) + (126*tmp27) + (882*x3)), None)
    tmp31 = tmp24 + tmp30
    tl.store(out_ptr0 + (x4), tmp31, None)
''')
