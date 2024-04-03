

# Original file: ./timm_vision_transformer___60.0/timm_vision_transformer___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/i5/ci5io3zsegs3dwjbfb2e75oqzyksgcc5yf56xvqht5g65fd4p32b.py
# Source Nodes: [add, add_1, getattr_l__mod___blocks___0___attn_proj_drop, getattr_l__mod___blocks___0___norm2], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add => add
# add_1 => add_3
# getattr_l__mod___blocks___0___attn_proj_drop => clone_1
# getattr_l__mod___blocks___0___norm2 => add_4, add_5, convert_element_type_2, convert_element_type_3, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
triton_per_fused_add_clone_native_layer_norm_7 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_7', '''
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
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 6304
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
    x0 = xindex % 197
    tmp0 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r2 + (384*x3)), rmask & xmask, other=0.0).to(tl.float32)
    tmp29 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tl.full([1], 384, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 / tmp14
    tmp16 = tmp6 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = tmp5 - tmp15
    tmp23 = 384.0
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
    tl.store(out_ptr2 + (r2 + (384*x3)), tmp35, rmask & xmask)
''')
