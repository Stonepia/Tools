

# Original file: ./mobilevit_s___60.0/mobilevit_s___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/ui/cuimmztm5inuismwnmfq2kfbbwpe5o5lhppwcts3pvgt5lsjyleu.py
# Source Nodes: [add_6, add_7, add_8, add_9, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___attn_proj_drop, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___mlp_drop2, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___attn_proj_drop, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___mlp_drop2, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___norm1], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add_6 => add_64
# add_7 => add_67
# add_8 => add_70
# add_9 => add_73
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___attn_proj_drop => clone_24
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___mlp_drop2 => clone_26
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___attn_proj_drop => clone_31
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___mlp_drop2 => clone_33
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___norm1 => add_74, add_75, convert_element_type_137, convert_element_type_138, mul_116, mul_117, rsqrt_9, sub_36, var_mean_9
triton_per_fused_add_clone_native_layer_norm_36 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_36', '''
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
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_36', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((192*(((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + ((x1 % 4) % 2)) % 16)) + (3072*((((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + (16*((x1 % 4) // 2)) + (32*((((4*x0) + (256*r2) + (49152*(x1 // 4)) + (x1 % 4)) // 32) % 98304)) + ((x1 % 4) % 2)) // 16) % 16)) + (49152*((((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + (16*((x1 % 4) // 2)) + (32*((((4*x0) + (256*r2) + (49152*(x1 // 4)) + (x1 % 4)) // 32) % 98304)) + ((x1 % 4) % 2)) // 49152) % 64)) + ((((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + (16*((x1 % 4) // 2)) + (32*((((4*x0) + (256*r2) + (49152*(x1 // 4)) + (x1 % 4)) // 32) % 98304)) + ((x1 % 4) % 2)) // 256) % 192)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r2 + (192*x3)), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r2 + (192*x3)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r2 + (192*x3)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_out_ptr0 + (r2 + (192*x3)), rmask, other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 / tmp18
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp9 - tmp19
    tmp27 = 192.0
    tmp28 = tmp25 / tmp27
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 * tmp34
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp35 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(in_out_ptr0 + (r2 + (192*x3)), tmp8, rmask)
    tl.store(out_ptr2 + (r2 + (192*x3)), tmp39, rmask)
''')