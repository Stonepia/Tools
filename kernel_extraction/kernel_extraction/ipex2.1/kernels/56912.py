

# Original file: ./twins_pcpvt_base___60.0/twins_pcpvt_base___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/w3/cw3djfa4ty3zgsnxgh42qqbu2qkiebl6hhqyecieyary4c5mrql5.py
# Source Nodes: [add_52, add_53, add_54, l__mod___blocks_3_1_attn_proj_drop, l__mod___blocks_3_1_mlp_drop2, l__mod___blocks_3_2_attn_proj_drop, l__mod___blocks_3_2_norm2], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add_52 => add_246
# add_53 => add_250
# add_54 => add_253
# l__mod___blocks_3_1_attn_proj_drop => clone_89
# l__mod___blocks_3_1_mlp_drop2 => clone_91
# l__mod___blocks_3_2_attn_proj_drop => clone_92
# l__mod___blocks_3_2_norm2 => add_254, add_255, convert_element_type_222, convert_element_type_223, mul_249, mul_250, rsqrt_84, sub_84, var_mean_84
triton_per_fused_add_clone_native_layer_norm_46 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_46', '''
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
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_46', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 3136
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp31 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp15 = tl.full([1], 512, tl.int32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 / tmp16
    tmp18 = tmp8 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tmp7 - tmp17
    tmp25 = 512.0
    tmp26 = tmp23 / tmp25
    tmp27 = 1e-06
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp24 * tmp29
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp30 * tmp32
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp33 + tmp35
    tmp37 = tmp36.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp37, rmask & xmask)
''')
