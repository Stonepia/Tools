

# Original file: ./hrnet_w18___60.0/hrnet_w18___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/24/c24m6a6q32vdnf4s56janyim5nlyti3eafz4mfvcrz3sc6evvgvv.py
# Source Nodes: [add_33, add_34, l__mod___stage4_0_fuse_act_2, l__mod___stage4_0_fuse_layers_2_3_2], Original ATen: [aten._unsafe_index, aten.add, aten.relu]
# add_33 => add_549
# add_34 => add_554
# l__mod___stage4_0_fuse_act_2 => relu_183
# l__mod___stage4_0_fuse_layers_2_3_2 => _unsafe_index_18
triton_poi_fused__unsafe_index_add_relu_8 = async_compile.triton('triton_poi_fused__unsafe_index_add_relu_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_relu_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__unsafe_index_add_relu_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x2 = (xindex // 1008) % 14
    x1 = (xindex // 72) % 14
    x0 = xindex % 72
    x3 = (xindex // 14112)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x4), None)
    tmp2 = tmp0 + tmp1
    tmp3 = x2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = 0.0
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10.to(tl.int32)
    tmp12 = x1
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp5
    tmp15 = tmp14 + tmp7
    tmp16 = tmp15 * tmp9
    tmp17 = tmp16.to(tl.int32)
    tmp18 = tl.load(in_ptr2 + (x0 + (72*tmp17) + (504*tmp11) + (3528*x3)), None)
    tmp19 = tmp2 + tmp18
    tmp20 = triton_helpers.maximum(0, tmp19)
    tl.store(out_ptr0 + (x4), tmp20, None)
''')
