

# Original file: ./hf_Bart___60.0/hf_Bart___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/on/conwb6qadpgzf4sdlpne6wpimzg7idxio5ckafxdjxtd726lspji.py
# Source Nodes: [add_13, add_18, dropout_24, dropout_27, l__mod___model_model_decoder_layers_0_self_attn_layer_norm, l__mod___model_model_encoder_layers_5_final_layer_norm], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add_13 => add_43
# add_18 => add_52
# dropout_24 => clone_48
# dropout_27 => clone_55
# l__mod___model_model_decoder_layers_0_self_attn_layer_norm => add_53, add_54, convert_element_type_55, convert_element_type_56, mul_55, mul_56, rsqrt_14, sub_21, var_mean_14
# l__mod___model_model_encoder_layers_5_final_layer_norm => add_44, add_45, convert_element_type_48, convert_element_type_49, mul_49, mul_50, rsqrt_12, sub_18, var_mean_12
triton_per_fused_add_clone_native_layer_norm_9 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_9', '''
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
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp20 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp45 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp48 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = tl.full([1], 768, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 / tmp12
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp22 = tmp20 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, 0)
    tmp27 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp31 = tmp30 / tmp12
    tmp32 = tmp24 - tmp31
    tmp33 = tmp32 * tmp32
    tmp34 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp36 = tl.where(rmask & xmask, tmp34, 0)
    tmp37 = triton_helpers.promote_to_tensor(tl.sum(tmp36, 0))
    tmp38 = tmp23 - tmp31
    tmp39 = 768.0
    tmp40 = tmp37 / tmp39
    tmp41 = 1e-05
    tmp42 = tmp40 + tmp41
    tmp43 = libdevice.rsqrt(tmp42)
    tmp44 = tmp38 * tmp43
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp44 * tmp46
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tmp47 + tmp49
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tmp3 - tmp13
    tmp53 = tmp19 / tmp39
    tmp54 = tmp53 + tmp41
    tmp55 = libdevice.rsqrt(tmp54)
    tmp56 = tmp52 * tmp55
    tmp57 = tmp56 * tmp46
    tmp58 = tmp57 + tmp49
    tmp59 = tmp58.to(tl.float32)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp51, rmask & xmask)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp59, rmask & xmask)
''')
