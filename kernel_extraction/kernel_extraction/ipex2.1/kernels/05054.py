

# Original file: ./swin_base_patch4_window7_224___60.0/swin_base_patch4_window7_224___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/d5/cd5jqi7ul32nuttw2n5gob3rnkzwxb2s5dxiwxoyupuxl65t7auk.py
# Source Nodes: [add_82, getattr_getattr_l__mod___layers___3___blocks___1___mlp_drop2, l__mod___norm], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add_82 => add_210
# getattr_getattr_l__mod___layers___3___blocks___1___mlp_drop2 => clone_263
# l__mod___norm => var_mean_52
triton_per_fused_add_clone_native_layer_norm_54 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_54', '''
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
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_54', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_54(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 3136
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 49
    x1 = (xindex // 49)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*(x0 % 7)) + (7168*((((7*(x0 // 7)) + (x0 % 7)) // 7) % 7)) + (50176*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (1024*(x0 % 7)) + (7168*((((7*(x0 // 7)) + (x0 % 7)) // 7) % 7)) + (50176*x1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (1024*x3)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r2 + (1024*x3)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_out_ptr0 + (r2 + (1024*x3)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 1024, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tl.store(in_out_ptr0 + (r2 + (1024*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr0 + (x3), tmp18, xmask)
    tl.store(out_ptr1 + (x3), tmp24, xmask)
''')
