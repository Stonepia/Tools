

# Original file: ./volo_d1_224___60.0/volo_d1_224___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/7u/c7uvnbuarw5hnlpnncsmjfwirm3yzsd7utzyzhqthcvfcvc7qqlo.py
# Source Nodes: [l__mod___post_network_0_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___post_network_0_norm1 => add_153, add_154, convert_element_type_169, convert_element_type_170, mul_153, mul_154, rsqrt_36, sub_57, var_mean_36
triton_per_fused_native_layer_norm_33 = async_compile.triton('triton_per_fused_native_layer_norm_33', '''
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
    size_hints=[16384, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_33', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_native_layer_norm_33(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
    xnumel = 12608
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp25 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, 0)
    tmp5 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp9 = tl.full([1], 384, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 / tmp10
    tmp12 = tmp2 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tmp1 - tmp11
    tmp19 = 384.0
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
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp31, rmask & xmask)
''')