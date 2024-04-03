

# Original file: ./hf_T5_generate__22_inference_62.2/hf_T5_generate__22_inference_62.2_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/uf/cufi6rhz3f3dtvzx6lurtxyacu67eqz636wqlhken2flphlu7drv.py
# Source Nodes: [add_11, add_13, add_15, add_16, l__self___decoder_block_1_layer__1__dropout, l__self___decoder_block_2_layer_0_self_attention_q, mean_6, mul_16, mul_17, pow_7, rsqrt_6], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_11 => add_14
# add_13 => add_17
# add_15 => add_19
# add_16 => add_20
# l__self___decoder_block_1_layer__1__dropout => clone_12
# l__self___decoder_block_2_layer_0_self_attention_q => convert_element_type_54
# mean_6 => mean_6
# mul_16 => mul_16
# mul_17 => mul_17
# pow_7 => pow_7
# rsqrt_6 => rsqrt_6
triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_12 = async_compile.triton('triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_12', '''
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
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
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
    tmp7 = tl.load(in_ptr3 + (r0), rmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr4 + (r0), rmask, other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 512.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp9 * tmp20
    tmp22 = tmp15 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp23, rmask)
''')
