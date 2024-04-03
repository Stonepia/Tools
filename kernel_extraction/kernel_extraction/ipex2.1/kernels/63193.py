

# Original file: ./MT5ForConditionalGeneration__0_backward_207.1/MT5ForConditionalGeneration__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/75/c75qya5wppc52nt3svq2b6hwnjbd4qnuqqzgb6ofs3376xdidle7.py
# Source Nodes: [add_56, add_59, l__self___decoder_dropout], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout, aten.native_dropout_backward, aten.pow, aten.sum]
# add_56 => add_66
# add_59 => add_70
# l__self___decoder_dropout => mul_148, mul_149
triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_25 = async_compile.triton('triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_25', '''
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
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp16', 7: '*fp16', 8: '*fp32', 9: '*i1', 10: '*fp16', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_25', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask)
    tmp9 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp24 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr8 + (r1 + (512*x0)), rmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 + tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = tmp6 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp26 = tmp6 * tmp25
    tmp27 = tmp24 + tmp26
    tmp28 = -0.5
    tmp29 = tmp23 * tmp28
    tmp30 = tmp25 * tmp25
    tmp31 = tmp30 * tmp25
    tmp32 = tmp29 * tmp31
    tmp33 = 512.0
    tmp34 = tmp32 / tmp33
    tmp35 = 2.0
    tmp36 = tmp18 * tmp35
    tmp37 = tmp34 * tmp36
    tmp38 = tmp27 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp41 * tmp11
    tmp43 = tmp39 * tmp42
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp38, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp43, rmask)
''')
