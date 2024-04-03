

# Original file: ./llama___60.0/llama___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/vh/cvhigoolpuofpgkuxt7lktwiy6zrpkawq2mish3wg65d5geuyyly.py
# Source Nodes: [add_15, add_17, add_19, add_21, add_22, float_30, mean_11, mul_39, mul_40, pow_12, rsqrt_11, type_as_30], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_15 => add_15
# add_17 => add_17
# add_19 => add_19
# add_21 => add_21
# add_22 => add_22
# float_30 => convert_element_type_69
# mean_11 => mean_11
# mul_39 => mul_44
# mul_40 => mul_45
# pow_12 => pow_12
# rsqrt_11 => rsqrt_11
# type_as_30 => convert_element_type_70
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_14 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_14', '''
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
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 1024
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp15 = 512.0
    tmp16 = tmp14 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp9 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp23 = tmp21 * tmp22
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp23, rmask & xmask)
''')
