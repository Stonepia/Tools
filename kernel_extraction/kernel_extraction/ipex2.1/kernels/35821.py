

# Original file: ./hf_T5_generate__42_inference_82.22/hf_T5_generate__42_inference_82.22_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/fv/cfvladmh4zmlqmvllfrux3h2zjun4e2gfujuc5e6frviyghklb6o.py
# Source Nodes: [add_15, add_18, add_19, l__self___decoder_block_1_layer_0_dropout, l__self___decoder_block_1_layer_1_dropout, mean_5, mul_14, mul_15, pow_6, rsqrt_5, to_14, to_15], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_15 => add_19
# add_18 => add_22
# add_19 => add_23
# l__self___decoder_block_1_layer_0_dropout => clone_8
# l__self___decoder_block_1_layer_1_dropout => clone_10
# mean_5 => mean_5
# mul_14 => mul_14
# mul_15 => mul_15
# pow_6 => pow_6
# rsqrt_5 => rsqrt_5
# to_14 => convert_element_type_23
# to_15 => convert_element_type_24
triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_13 = async_compile.triton('triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_13', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r0), rmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (r0), rmask, other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = 512.0
    tmp13 = tmp10 / tmp12
    tmp14 = 1e-06
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp5 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp11 * tmp18
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp19, rmask)
''')