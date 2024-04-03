

# Original file: ./tnt_s_patch16_224___60.0/tnt_s_patch16_224___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/f3/cf3aws2kqmnhbobnyalayuxphhz6qehzetyuye3zzbwbmxuym5du.py
# Source Nodes: [l__self___blocks_0_attn_in_qk, l__self___blocks_0_norm_in, reshape_2], Original ATen: [aten._to_copy, aten.clone, aten.native_layer_norm]
# l__self___blocks_0_attn_in_qk => convert_element_type_11
# l__self___blocks_0_norm_in => add_8, add_9, mul_4, mul_5, rsqrt_2, sub_2, var_mean_2
# reshape_2 => clone_2
triton_per_fused__to_copy_clone_native_layer_norm_3 = async_compile.triton('triton_per_fused__to_copy_clone_native_layer_norm_3', '''
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
    size_hints=[524288, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_clone_native_layer_norm_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_clone_native_layer_norm_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 16
    x1 = (xindex // 16)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((24*(x0 % 4)) + (24*(tl.where(((4*((x1 % 196) % 14)) + (x0 % 4)) >= 0, 0, 56))) + (96*((x1 % 196) % 14)) + (1344*((((4*(x0 // 4)) + (x0 % 4)) // 4) % 4)) + (1344*(tl.where(((4*((x1 % 196) // 14)) + ((((4*(x0 // 4)) + (x0 % 4)) // 4) % 4)) >= 0, 0, 56))) + (5376*((x1 % 196) // 14)) + (75264*(x1 // 196)) + ((((4*(x0 // 4)) + (16*r2) + (x0 % 4)) // 16) % 24)), rmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x0 + (16*r2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tl.full([XBLOCK, 1], 24, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 / tmp14
    tmp16 = tmp6 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = tmp5 - tmp15
    tmp23 = 24.0
    tmp24 = tmp21 / tmp23
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp32.to(tl.float32)
    tl.store(out_ptr2 + (r2 + (24*x3)), tmp33, rmask)
''')
