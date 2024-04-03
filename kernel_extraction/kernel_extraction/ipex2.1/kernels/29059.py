

# Original file: ./mobilevit_s___60.0/mobilevit_s___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/37/c37cgs3a6z4gbpwpoqrerifqppdagvasw6o2i5h6mbvuwmover7v.py
# Source Nodes: [add_10, add_11, add_12, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___attn_proj_drop, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___mlp_drop2, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___attn_proj_drop, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___norm2], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add_10 => add_76
# add_11 => add_79
# add_12 => add_82
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___attn_proj_drop => clone_38
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___mlp_drop2 => clone_40
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___attn_proj_drop => clone_45
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___norm2 => add_83, add_84, convert_element_type_149, convert_element_type_150, mul_127, mul_128, rsqrt_12, sub_41, var_mean_12
triton_per_fused_add_clone_native_layer_norm_39 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_39', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_39', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (192*x0)), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1 + (192*x0)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r1 + (192*x0)), rmask, other=0.0).to(tl.float32)
    tmp31 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 / tmp16
    tmp18 = tmp8 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = tmp7 - tmp17
    tmp25 = 192.0
    tmp26 = tmp23 / tmp25
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp24 * tmp29
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp30 * tmp32
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp33 + tmp35
    tmp37 = tmp36.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (192*x0)), tmp37, rmask)
''')
