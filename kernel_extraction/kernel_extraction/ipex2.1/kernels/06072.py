

# Original file: ./crossvit_9_240___60.0/crossvit_9_240___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/jf/cjfufmcsjmo2cptbknp7v3iak5eb3g7qzqsz7hxoavuy4nkn44j7.py
# Source Nodes: [l__self___blocks_1_projs_0_0, l__self___blocks_1_projs_0_1, l__self___blocks_1_projs_0_2], Original ATen: [aten._to_copy, aten.gelu, aten.native_layer_norm]
# l__self___blocks_1_projs_0_0 => add_122, add_123, clone_44, mul_168, mul_169, rsqrt_22, sub_72, var_mean_22
# l__self___blocks_1_projs_0_1 => add_124, erf_12, mul_170, mul_171, mul_172
# l__self___blocks_1_projs_0_2 => convert_element_type_152
triton_per_fused__to_copy_gelu_native_layer_norm_53 = async_compile.triton('triton_per_fused__to_copy_gelu_native_layer_norm_53', '''
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
    size_hints=[128, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_gelu_native_layer_norm_53', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_gelu_native_layer_norm_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (r1 + (51328*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (51328*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r1 + (51328*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = tmp6 - tmp16
    tmp24 = 128.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-06
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = 0.5
    tmp35 = tmp33 * tmp34
    tmp36 = 0.7071067811865476
    tmp37 = tmp33 * tmp36
    tmp38 = libdevice.erf(tmp37)
    tmp39 = 1.0
    tmp40 = tmp38 + tmp39
    tmp41 = tmp35 * tmp40
    tmp42 = tmp41.to(tl.float32)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp42, rmask & xmask)
''')
