

# Original file: ./crossvit_9_240___60.0/crossvit_9_240___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/tr/ctrkn72fbotfexiemnj5d7jkj45lhl3fabfeellq27rdaxevedzz.py
# Source Nodes: [add_11, getitem_22, l__self___blocks_0_fusion_1_attn_proj_drop, l__self___blocks_0_revert_projs_1_0, l__self___blocks_0_revert_projs_1_1, l__self___blocks_0_revert_projs_1_2], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.gelu, aten.native_layer_norm, aten.slice]
# add_11 => add_90
# getitem_22 => slice_20
# l__self___blocks_0_fusion_1_attn_proj_drop => clone_27
# l__self___blocks_0_revert_projs_1_0 => add_91, add_92, mul_133, mul_134, rsqrt_13, sub_62, var_mean_13
# l__self___blocks_0_revert_projs_1_1 => add_93, erf_7, mul_135, mul_136, mul_137
# l__self___blocks_0_revert_projs_1_2 => convert_element_type_98
triton_per_fused__to_copy_add_clone_gelu_native_layer_norm_slice_47 = async_compile.triton('triton_per_fused__to_copy_add_clone_gelu_native_layer_norm_slice_47', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_gelu_native_layer_norm_slice_47', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_gelu_native_layer_norm_slice_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 / tmp12
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = tmp3 - tmp13
    tmp21 = 128.0
    tmp22 = tmp19 / tmp21
    tmp23 = 1e-06
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = 0.5
    tmp32 = tmp30 * tmp31
    tmp33 = 0.7071067811865476
    tmp34 = tmp30 * tmp33
    tmp35 = libdevice.erf(tmp34)
    tmp36 = 1.0
    tmp37 = tmp35 + tmp36
    tmp38 = tmp32 * tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp39, rmask & xmask)
''')