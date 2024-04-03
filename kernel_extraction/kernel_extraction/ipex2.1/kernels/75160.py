

# Original file: ./detectron2_maskrcnn_r_50_c4__51_inference_91.31/detectron2_maskrcnn_r_50_c4__51_inference_91.31_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/l2/cl23daks2q3an7ibt2cpvoinw4kq5xh6vgk74pii3pqlzvh4vcuc.py
# Source Nodes: [add_1, max_1], Original ATen: [aten.add, aten.max]
# add_1 => add_1
# max_1 => max_1
triton_per_fused_add_max_0 = async_compile.triton('triton_per_fused_add_max_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[1, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_max_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_max_0(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 972
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    r2 = (rindex // 4)
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (2*r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp3, 0))
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1.0
    tmp8 = tmp4 + tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp0 + tmp9
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp10, rmask)
''')
