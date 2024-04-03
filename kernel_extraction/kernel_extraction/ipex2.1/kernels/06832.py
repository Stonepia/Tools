

# Original file: ./DistilBertForQuestionAnswering__0_forward_97.0/DistilBertForQuestionAnswering__0_forward_97.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/un/cuneaczg6fzhe3yexmstp637qeuvms6ts4xqe4n5wzgjmqqkuhny.py
# Source Nodes: [add_1, l__mod___distilbert_transformer_layer_0_sa_layer_norm], Original ATen: [aten.add, aten.native_layer_norm]
# add_1 => add_3
# l__mod___distilbert_transformer_layer_0_sa_layer_norm => add_4, add_5, convert_element_type_5, convert_element_type_6, mul_6, mul_7, rsqrt_1, sub_2, var_mean_1
triton_per_fused_add_native_layer_norm_8 = async_compile.triton('triton_per_fused_add_native_layer_norm_8', '''
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
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_8(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 32768
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = tl.full([1], 768, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 / tmp12
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = 768.0
    tmp21 = tmp19 / tmp20
    tmp22 = 1e-12
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp3 - tmp13
    tmp26 = tmp25 * tmp24
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 * tmp28
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 + tmp31
    tmp33 = tmp32.to(tl.float32)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp2, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp24, None)
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp33, rmask)
    tl.store(out_ptr0 + (x0), tmp13, None)
''')
