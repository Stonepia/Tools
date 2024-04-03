

# Original file: ./T5ForConditionalGeneration__0_forward_169.0/T5ForConditionalGeneration__0_forward_169.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/7d/c7dpebpmagb27nxirtsgjrtnpyykrwtqvuc4jym4fc34dwasj7x2.py
# Source Nodes: [add_63, add_65, add_67, add_68, l__self___decoder_block_5_layer__1__dropout, l__self___decoder_dropout_1, l__self___lm_head, mean_31, mul_69, mul_70, mul_71, pow_32, rsqrt_31], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.native_dropout, aten.pow, aten.rsqrt, aten.view]
# add_63 => add_81
# add_65 => add_84
# add_67 => add_86
# add_68 => add_87
# l__self___decoder_block_5_layer__1__dropout => gt_63, mul_193, mul_194
# l__self___decoder_dropout_1 => gt_64, mul_197, mul_198
# l__self___lm_head => convert_element_type_247, view_428
# mean_31 => mean_31
# mul_69 => mul_195
# mul_70 => mul_196
# mul_71 => mul_199
# pow_32 => pow_32
# rsqrt_31 => rsqrt_31
triton_per_fused__to_copy_add_mean_mul_native_dropout_pow_rsqrt_view_21 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_native_dropout_pow_rsqrt_view_21', '''
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
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*bf16', 5: '*bf16', 6: '*fp32', 7: '*i1', 8: '*i1', 9: '*bf16', 10: 'i32', 11: 'i32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_native_dropout_pow_rsqrt_view_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_native_dropout_pow_rsqrt_view_21(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr3, out_ptr4, load_seed_offset, load_seed_offset1, xnumel, rnumel):
    xnumel = 4096
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
    tmp7 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp34 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (512*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 + tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 + tmp16
    tmp18 = tmp10.to(tl.float32)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = 512.0
    tmp26 = tmp24 / tmp25
    tmp27 = 1e-06
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tl.load(in_ptr0 + load_seed_offset1)
    tmp31 = tl.rand(tmp30, (tmp1).to(tl.uint32))
    tmp32 = tmp31 > tmp4
    tmp33 = tmp32.to(tl.float32)
    tmp35 = tmp19 * tmp29
    tmp36 = tmp34 * tmp35
    tmp37 = tmp33 * tmp36
    tmp38 = tmp37 * tmp9
    tmp39 = 0.04419417382415922
    tmp40 = tmp38 * tmp39
    tmp41 = tmp40.to(tl.float32)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp5, rmask)
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp10, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp29, None)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp32, rmask)
    tl.store(out_ptr4 + (r1 + (512*x0)), tmp41, rmask)
''')
