

# Original file: ./swin_base_patch4_window7_224___60.0/swin_base_patch4_window7_224___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/6k/c6kfhqwlv2qtd5gl2cajhogjdy6vvi6gu7qzyyn6i7rocbgy7bid.py
# Source Nodes: [contiguous, getattr_getattr_l__mod___layers___0___blocks___0___norm1, l__mod___patch_embed_norm], Original ATen: [aten.clone, aten.native_layer_norm]
# contiguous => clone_1
# getattr_getattr_l__mod___layers___0___blocks___0___norm1 => convert_element_type_2, var_mean_1
# l__mod___patch_embed_norm => add, add_1, clone, convert_element_type, convert_element_type_1, mul, mul_1, rsqrt, sub, var_mean
triton_per_fused_clone_native_layer_norm_1 = async_compile.triton('triton_per_fused_clone_native_layer_norm_1', '''
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
    size_hints=[262144, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_native_layer_norm_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_clone_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 7
    x3 = (xindex // 7) % 8
    x4 = (xindex // 56) % 7
    x5 = (xindex // 392)
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0).to(tl.float32)
    tmp25 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp52 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp55 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask, tmp2, 0)
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 / tmp10
    tmp12 = tmp2 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = tmp1 - tmp11
    tmp19 = 128.0
    tmp20 = tmp17 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 * tmp26
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
    tmp35 = tl.where(rmask, tmp33, 0)
    tmp36 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
    tmp38 = tl.where(rmask, tmp36, 0)
    tmp39 = tl.sum(tmp38, 1)[:, None]
    tmp40 = tmp39 / tmp10
    tmp41 = tmp33 - tmp40
    tmp42 = tmp41 * tmp41
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK, RBLOCK])
    tmp45 = tl.where(rmask, tmp43, 0)
    tmp46 = tl.sum(tmp45, 1)[:, None]
    tmp47 = tmp32 - tmp40
    tmp48 = tmp46 / tmp19
    tmp49 = tmp48 + tmp21
    tmp50 = libdevice.rsqrt(tmp49)
    tmp51 = tmp47 * tmp50
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp51 * tmp53
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tmp54 + tmp56
    tmp58 = tmp57.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp31, rmask)
    tl.store(out_ptr5 + (r1 + (128*x2) + (896*x4) + (6272*x3) + (50176*x5)), tmp58, rmask)
''')
