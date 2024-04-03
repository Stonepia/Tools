

# Original file: ./gmixer_24_224___60.0/gmixer_24_224___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/3g/c3gztoyz3fysow7ujhnqvrzaxygarxb73v4te45krcr24xpzfm72.py
# Source Nodes: [add, add_1, add_2, add_3, getattr_l__mod___blocks___0___mlp_channels_drop2, getattr_l__mod___blocks___1___mlp_channels_drop2, getattr_l__mod___blocks___2___norm1], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add => add_3
# add_1 => add_6
# add_2 => add_10
# add_3 => add_13
# getattr_l__mod___blocks___0___mlp_channels_drop2 => clone_6
# getattr_l__mod___blocks___1___mlp_channels_drop2 => clone_13
# getattr_l__mod___blocks___2___norm1 => clone_14, var_mean_4
triton_per_fused_add_clone_native_layer_norm_9 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_9', '''
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
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 25088
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (196*r2) + (75264*x1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (384*x3)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (x0 + (196*r2) + (75264*x1)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_out_ptr0 + (r2 + (384*x3)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 384, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tl.store(in_out_ptr0 + (r2 + (384*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr0 + (x3), tmp18, xmask)
    tl.store(out_ptr1 + (x3), tmp24, xmask)
''')
