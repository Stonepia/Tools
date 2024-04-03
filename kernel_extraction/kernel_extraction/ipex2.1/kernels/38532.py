

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/n3/cn3qvkhrqtnpwh7nx5lrm55hn2jpgkenq2h5otq74pqblfnuhqre.py
# Source Nodes: [grid_sample_4, l1_loss_fn_3], Original ATen: [aten.abs, aten.grid_sampler_2d, aten.mean, aten.sub]
# grid_sample_4 => add_123, index_27, mul_222
# l1_loss_fn_3 => abs_4, mean_4, sub_123
triton_red_fused_abs_grid_sampler_2d_mean_sub_51 = async_compile.triton('triton_red_fused_abs_grid_sampler_2d_mean_sub_51', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_grid_sampler_2d_mean_sub_51', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_abs_grid_sampler_2d_mean_sub_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 273
    rnumel = 8170
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = r1 + (8170*x0)
        tmp1 = tl.full([1, 1], 2230272, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((r1 + (8170*x0)) % 2230272), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((123904*(((r1 + (8170*x0)) // 371712) % 6)) + ((r1 + (8170*x0)) % 123904)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.where(tmp4 < 0, tmp4 + 352, tmp4)
        # tl.device_assert(((0 <= tmp5) & (tmp5 < 352)) | ~(tmp2 & rmask & xmask), "index out of bounds: 0 <= tmp5 < 352")
        tmp6 = tl.load(in_ptr2 + ((123904*(((r1 + (8170*x0)) // 371712) % 6)) + ((r1 + (8170*x0)) % 123904)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.where(tmp6 < 0, tmp6 + 352, tmp6)
        # tl.device_assert(((0 <= tmp7) & (tmp7 < 352)) | ~(tmp2 & rmask & xmask), "index out of bounds: 0 <= tmp7 < 352")
        tmp8 = tl.load(in_ptr3 + (tmp7 + (352*tmp5) + (123904*(((r1 + (8170*x0)) // 123904) % 18)) + (123904*(tl.where((((r1 + (8170*x0)) // 123904) % 3) >= 0, 0, 3))) + (371712*(tl.where((((r1 + (8170*x0)) // 371712) % 6) >= 0, 0, 6)))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr4 + ((123904*(((r1 + (8170*x0)) // 371712) % 6)) + ((r1 + (8170*x0)) % 123904)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tmp8 * tmp9
        tmp11 = tmp3 + tmp10
        tmp12 = tl.load(in_ptr5 + ((r1 + (8170*x0)) % 2230272), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 - tmp12
        tmp14 = tl.abs(tmp13)
        tmp15 = tl.where(tmp2, tmp14, 0)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp17, xmask)
''')
