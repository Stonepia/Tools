

# Original file: ./swin_base_patch4_window7_224___60.0/swin_base_patch4_window7_224___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/4e/c4exahwdpm5lnqli6lebxunt6dz5lyjc6dashgeqkn7jt7d7kb4r.py
# Source Nodes: [contiguous_4], Original ATen: [aten.clone]
# contiguous_4 => clone_12
triton_poi_fused_clone_10 = async_compile.triton('triton_poi_fused_clone_10', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*bf16', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x5 = (xindex // 401408)
    x6 = (xindex // 128) % 56
    x7 = (xindex // 7168) % 56
    x1 = (xindex // 128) % 7
    x2 = (xindex // 896) % 8
    x3 = (xindex // 7168) % 7
    x4 = (xindex // 50176) % 8
    x8 = xindex % 896
    x9 = (xindex // 50176)
    tmp0 = tl.load(in_ptr0 + (x0 + (128*((3 + x6) % 56)) + (7168*((((56*((3 + x7) % 56)) + ((3 + x6) % 56)) // 56) % 56)) + (401408*x5)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*(((3 + x1 + (7*x2)) % 56) % 7)) + (896*(((((56*((3 + x3 + (7*x4)) % 56)) + ((3 + x1 + (7*x2)) % 56)) // 56) % 56) % 7)) + (6272*(((3 + x1 + (7*x2)) % 56) // 7)) + (50176*(((((56*((3 + x3 + (7*x4)) % 56)) + ((3 + x1 + (7*x2)) % 56)) // 56) % 56) // 7)) + (401408*x5)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + (128*((3 + x6) % 56)) + (7168*((3 + x7) % 56)) + (401408*x5)), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + ((56*((3 + x7) % 56)) + (3136*x5) + ((3 + x6) % 56)), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + ((56*((3 + x7) % 56)) + (3136*x5) + ((3 + x6) % 56)), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 - tmp6
    tmp9 = 128.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp7 * tmp13
    tmp16 = tmp14 * tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.store(out_ptr0 + (x8 + (896*x3) + (6272*x2) + (50176*x9)), tmp19, None)
''')
