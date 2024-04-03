

# Original file: ./sam___60.0/sam___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/k6/ck6xykov73czztwjaoq7kp6pl2gdqhl4t4d5pc4xbfi3myrvgalh.py
# Source Nodes: [add_197, add_199, l__self___mask_decoder_transformer_layers_0_cross_attn_token_to_image_k_proj, l__self___mask_decoder_transformer_layers_0_cross_attn_token_to_image_v_proj, mean_2, mean_3, pow_2, sub_67], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.pow, aten.sub]
# add_197 => add_357
# add_199 => add_361
# l__self___mask_decoder_transformer_layers_0_cross_attn_token_to_image_k_proj => convert_element_type_605
# l__self___mask_decoder_transformer_layers_0_cross_attn_token_to_image_v_proj => convert_element_type_608
# mean_2 => mean_2
# mean_3 => mean_3
# pow_2 => convert_element_type_584, pow_2
# sub_67 => sub_163
triton_per_fused__to_copy_add_mean_pow_sub_46 = async_compile.triton('triton_per_fused__to_copy_add_mean_pow_sub_46', '''
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
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: '*fp32', 6: '*bf16', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_pow_sub_46', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_pow_sub_46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.load(in_ptr4 + (r1 + (256*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.where(rmask, tmp2, 0)
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp6 = 256.0
    tmp7 = tmp5 / tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp0 - tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = tmp15 / tmp6
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = tmp10 / tmp20
    tmp22 = tmp16 * tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 + tmp25
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp26.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp26, rmask)
    tl.store(out_ptr3 + (r1 + (256*x0)), tmp30, rmask)
    tl.store(out_ptr4 + (r1 + (256*x0)), tmp31, rmask)
''')
