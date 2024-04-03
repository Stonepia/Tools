

# Original file: ./hrnet_w18___60.0/hrnet_w18___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/5m/c5mpsivdgcd67v5svvywhi66hkxljfuw2suhaxcka2der6yzaup5.py
# Source Nodes: [add_30, add_31, l__mod___stage4_0_fuse_act_1, l__mod___stage4_0_fuse_layers_1_2_2, l__mod___stage4_0_fuse_layers_1_3_2], Original ATen: [aten._unsafe_index, aten.add, aten.relu]
# add_30 => add_536
# add_31 => add_541
# l__mod___stage4_0_fuse_act_1 => relu_181
# l__mod___stage4_0_fuse_layers_1_2_2 => _unsafe_index_16
# l__mod___stage4_0_fuse_layers_1_3_2 => _unsafe_index_17
triton_poi_fused__unsafe_index_add_relu_7 = async_compile.triton('triton_poi_fused__unsafe_index_add_relu_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_relu_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__unsafe_index_add_relu_7(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x2 = (xindex // 1008) % 28
    x1 = (xindex // 36) % 28
    x0 = xindex % 36
    x3 = (xindex // 28224)
    tmp0 = tl.load(in_out_ptr0 + (x4), None)
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
    tmp16 = tl.load(in_ptr0 + (x0 + (36*tmp15) + (504*tmp9) + (7056*x3)), None)
    tmp17 = tmp0 + tmp16
    tmp18 = 0.25
    tmp19 = tmp6 * tmp18
    tmp20 = tmp19.to(tl.int32)
    tmp21 = tmp13 * tmp18
    tmp22 = tmp21.to(tl.int32)
    tmp23 = tl.load(in_ptr1 + (x0 + (36*tmp22) + (252*tmp20) + (1764*x3)), None)
    tmp24 = tmp17 + tmp23
    tmp25 = triton_helpers.maximum(0, tmp24)
    tl.store(in_out_ptr0 + (x4), tmp25, None)
''')
