

# Original file: ./levit_128___60.0/levit_128___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/q5/cq5yfzsuxi3hhlnl6m3k3srluoewi5iyzf2w76e52v2t7exjjkhw.py
# Source Nodes: [matmul_21], Original ATen: [aten.clone]
# matmul_21 => clone_54
triton_poi_fused_clone_39 = async_compile.triton('triton_poi_fused_clone_39', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_39', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 16
    x2 = (xindex // 512) % 12
    x3 = (xindex // 6144)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (32 + x0 + (64*x2) + (768*x1) + (12288*x3)), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp9.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp10, None)
''')
