

# Original file: ./hf_Reformer__25_inference_65.5/hf_Reformer__25_inference_65.5_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/sq/csqudzoukk2s42z5pd5cf37fz6jpvuryvuw7bg6rrdw75cxs3rze.py
# Source Nodes: [add_4, cat_68, cat_70, gather, mean, mul_2, pow_1, rsqrt, scatter_, truediv_1, wrapped_sqrt], Original ATen: [aten._to_copy, aten.add, aten.cat, aten.div, aten.gather, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.scatter, aten.sqrt]
# add_4 => add_11
# cat_68 => cat_13
# cat_70 => cat_11
# gather => gather
# mean => mean
# mul_2 => mul_8
# pow_1 => convert_element_type_20, pow_1
# rsqrt => rsqrt_3
# scatter_ => scatter
# truediv_1 => div_1
# wrapped_sqrt => full_default
triton_per_fused__to_copy_add_cat_div_gather_mean_mul_pow_rsqrt_scatter_sqrt_23 = async_compile.triton('triton_per_fused__to_copy_add_cat_div_gather_mean_mul_pow_rsqrt_scatter_sqrt_23', '''
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
    size_hints=[65536, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*i64', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp16', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_cat_div_gather_mean_mul_pow_rsqrt_scatter_sqrt_23', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_cat_div_gather_mean_mul_pow_rsqrt_scatter_sqrt_23(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    x2 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    x4 = xindex % 4096
    x5 = (xindex // 4096)
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], 4096, tl.int64)
    tmp2 = tmp0 % tmp1
    tmp3 = tmp2 + tmp1
    tmp4 = tl.where(((tmp2 != 0) & ((tmp2 < 0) != (tmp1 < 0))), tmp3, tmp2)
    tmp5 = tl.where(tmp0 < 0, tmp0 + 4096, tmp0)
    # tl.device_assert((0 <= tmp5) & (tmp5 < 4096), "index out of bounds: 0 <= tmp5 < 4096")
    tmp6 = x4
    tmp7 = tl.where(tmp4 < 0, tmp4 + 4096, tmp4)
    # tl.device_assert((0 <= tmp7) & (tmp7 < 4096), "index out of bounds: 0 <= tmp7 < 4096")
    tmp8 = tl.load(in_ptr1 + (r3 + (64*x5) + (768*tmp7)), rmask, other=0.0).to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = 64.0
    tmp16 = tmp14 / tmp15
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp9 * tmp19
    tmp21 = 8.0
    tmp22 = tmp20 / tmp21
    tmp23 = tl.load(in_ptr2 + (r3 + (64*x5) + (768*tmp7)), rmask, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (128*x1)), tmp4, None)
    tl.store(out_ptr1 + (tmp5 + (4096*x5)), tmp6, None)
    tl.store(out_ptr3 + (r3 + (64*x0) + (8192*x1)), tmp22, rmask)
    tl.store(out_ptr4 + (r3 + (64*x2)), tmp8, rmask)
    tl.store(out_ptr5 + (r3 + (64*x0) + (8192*x1)), tmp23, rmask)
    tl.store(out_ptr2 + (x2), tmp14, None)
''')
