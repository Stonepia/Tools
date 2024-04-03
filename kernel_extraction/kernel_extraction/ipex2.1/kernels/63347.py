

# Original file: ./MT5ForConditionalGeneration__0_backward_207.1/MT5ForConditionalGeneration__0_backward_207.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/6u/c6u2w4nqj4npmnbxk6vrbtuvyr2ku73l7xompwj7f3mgh2dmkvev.py
# Source Nodes: [to_87], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
# to_87 => convert_element_type_137
triton_per_fused__to_copy_add_div_mul_native_dropout_backward_pow_sum_8 = async_compile.triton('triton_per_fused__to_copy_add_div_mul_native_dropout_backward_pow_sum_8', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: '*i1', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_native_dropout_backward_pow_sum_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_native_dropout_backward_pow_sum_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = tmp5 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp13 + tmp16
    tmp18 = -0.5
    tmp19 = tmp12 * tmp18
    tmp20 = tmp14 * tmp14
    tmp21 = tmp20 * tmp14
    tmp22 = tmp19 * tmp21
    tmp23 = 512.0
    tmp24 = tmp22 / tmp23
    tmp25 = 2.0
    tmp26 = tmp7 * tmp25
    tmp27 = tmp24 * tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp17 + tmp28
    tmp31 = tmp30.to(tl.float32)
    tmp32 = 1.1111111111111112
    tmp33 = tmp31 * tmp32
    tmp34 = tmp29 * tmp33
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp29, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp34, rmask)
''')
