

# Original file: ./poolformer_m36___60.0/poolformer_m36___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/xq/cxqbwlvnbi4tp55le3ah4vjg4eahbbmlwxcr5ihclk2d7fn2xsiv.py
# Source Nodes: [group_norm_1], Original ATen: [aten.native_group_norm]
# group_norm_1 => convert_element_type_4, var_mean_1
triton_per_fused_native_group_norm_7 = async_compile.triton('triton_per_fused_native_group_norm_7', '''
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
    size_hints=[65536, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_native_group_norm_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 57344
    XBLOCK: tl.constexpr = 1
    rnumel = 336
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 4
    x1 = (xindex // 4) % 224
    x2 = (xindex // 896)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((24*x0) + (96*((r3 + (336*x1)) % 3136)) + (301056*x2) + ((r3 + (336*x1)) // 3136)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + ((24*x0) + (96*((r3 + (336*x1)) % 3136)) + (301056*x2) + ((r3 + (336*x1)) // 3136)), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + ((24*x0) + (96*((r3 + (336*x1)) % 3136)) + (301056*x2) + ((r3 + (336*x1)) // 3136)), rmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + ((24*x0) + ((r3 + (336*x1)) // 3136)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp15 = tl.full([1], 336, tl.int32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 / tmp16
    tmp18 = tmp8 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tl.store(out_ptr0 + (x4), tmp17, None)
    tl.store(out_ptr1 + (x4), tmp23, None)
    tl.store(out_ptr2 + (x4), tmp16, None)
''')