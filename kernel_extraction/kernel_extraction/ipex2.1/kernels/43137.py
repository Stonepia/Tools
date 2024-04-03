

# Original file: ./speech_transformer__28_inference_68.8/speech_transformer__28_inference_68.8_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/63/c63oeklod65oddboumgrio7hhspbuqxfv25wlmroihnokdfsmn3o.py
# Source Nodes: [add, add_1, imul, l__self___layer_stack_0_enc_attn_w_qs, l__self___layer_stack_0_slf_attn_dropout, l__self___layer_stack_0_slf_attn_layer_norm, l__self___tgt_word_emb, mul], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.embedding, aten.mul, aten.native_layer_norm]
# add => add
# add_1 => add_1
# imul => mul_3
# l__self___layer_stack_0_enc_attn_w_qs => convert_element_type_13
# l__self___layer_stack_0_slf_attn_dropout => clone_6
# l__self___layer_stack_0_slf_attn_layer_norm => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
# l__self___tgt_word_emb => embedding
# mul => mul
triton_per_fused__to_copy_add_clone_embedding_mul_native_layer_norm_4 = async_compile.triton('triton_per_fused__to_copy_add_clone_embedding_mul_native_layer_norm_4', '''
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
    size_hints=[256, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_embedding_mul_native_layer_norm_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_embedding_mul_native_layer_norm_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 220
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 22
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (r2 + (512*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.where(tmp2 < 0, tmp2 + 1014, tmp2)
    # tl.device_assert(((0 <= tmp3) & (tmp3 < 1014)) | ~xmask, "index out of bounds: 0 <= tmp3 < 1014")
    tmp4 = tl.load(in_ptr2 + (r2 + (512*tmp3)), rmask & xmask, other=0.0)
    tmp5 = 22.627416997969522
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp1 + tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tl.full([1], 512, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 / tmp18
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tmp9 - tmp19
    tmp27 = 512.0
    tmp28 = tmp25 / tmp27
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tmp38 = tmp36 * tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp39, rmask & xmask)
''')
