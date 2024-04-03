

# Original file: ./hf_T5_generate__70_inference_110.50/hf_T5_generate__70_inference_110.50.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/vp/cvpgmzfzp2ai2z4tzx6eqr4i2xgojwsp6vldiy5dvok2b76wupcq.py
# Source Nodes: [add_15, add_18, add_19, l__self___decoder_block_1_layer__1__dense_relu_dense_wi, mean_5, mul_14, mul_15, pow_6, rsqrt_5], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_15 => add_19
# add_18 => add_22
# add_19 => add_23
# l__self___decoder_block_1_layer__1__dense_relu_dense_wi => convert_element_type_41
# mean_5 => mean_5
# mul_14 => mul_14
# mul_15 => mul_15
# pow_6 => pow_6
# rsqrt_5 => rsqrt_5
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13', '''
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
    size_hints=[1, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0), rmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (r0), rmask, other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = 512.0
    tmp14 = tmp11 / tmp13
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp6 * tmp17
    tmp19 = tmp12 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp20, rmask)
''')
