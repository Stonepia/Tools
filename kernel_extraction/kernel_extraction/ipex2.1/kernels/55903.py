

# Original file: ./cait_m36_384___60.0/cait_m36_384___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/34/c34yfk34u4w3f5s74nfsxjvflketndaomoohpxxjzduiqr3epruz.py
# Source Nodes: [add_73, l__mod___blocks_token_only_0_attn_proj_drop, l__mod___blocks_token_only_0_norm2, mul_108], Original ATen: [aten.add, aten.clone, aten.mul, aten.native_layer_norm]
# add_73 => add_327
# l__mod___blocks_token_only_0_attn_proj_drop => clone_289
# l__mod___blocks_token_only_0_norm2 => add_328, add_329, convert_element_type_292, convert_element_type_293, mul_365, mul_366, rsqrt_73, sub_110, var_mean_73
# mul_108 => mul_364
triton_per_fused_add_clone_mul_native_layer_norm_20 = async_compile.triton('triton_per_fused_add_clone_mul_native_layer_norm_20', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_mul_native_layer_norm_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_mul_native_layer_norm_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (r0), rmask, other=0.0).to(tl.float32)
    tmp29 = tl.load(in_ptr3 + (r0), rmask, other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr4 + (r0), rmask, other=0.0).to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tl.full([1], 768, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 / tmp14
    tmp16 = tmp6 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = tmp5 - tmp15
    tmp23 = 768.0
    tmp24 = tmp21 / tmp23
    tmp25 = 1e-06
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 * tmp30
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr2 + (tl.broadcast_to(r0, [RBLOCK])), tmp35, rmask)
''')
