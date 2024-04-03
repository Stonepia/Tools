

# Original file: ./coat_lite_mini___60.0/coat_lite_mini___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/cw/ccwzhtbt3iluz3yffvp6c56xwm4lbqpfnn4ti3o7f4jouqieozus.py
# Source Nodes: [l__self___head, l__self___head_drop], Original ATen: [aten._to_copy, aten.clone]
# l__self___head => convert_element_type_208
# l__self___head_drop => clone_64
triton_poi_fused__to_copy_clone_88 = async_compile.triton('triton_poi_fused__to_copy_clone_88', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*bf16', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_88', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_clone_88(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (25600*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (25600*x1)), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x0 + (25600*x1)), None).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (50*x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (50*x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 512.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp20, None)
''')
