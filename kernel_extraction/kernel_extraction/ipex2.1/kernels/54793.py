

# Original file: ./AllenaiLongformerBase__22_forward_137.2/AllenaiLongformerBase__22_forward_137.2_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/z5/cz5xxs3yngkziuabytcaoclzuriyjfarvjjfsuv4qx5aak7aqa7g.py
# Source Nodes: [iadd, softmax], Original ATen: [aten._softmax, aten.add]
# iadd => add_5
# softmax => amax, clone_5, div_7, exp, sub_4, sum_1
triton_per_fused__softmax_add_20 = async_compile.triton('triton_per_fused__softmax_add_20', '''
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
    size_hints=[65536, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_add_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, xnumel, rnumel):
    xnumel = 49152
    XBLOCK: tl.constexpr = 1
    rnumel = 513
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x1 = (xindex // 12) % 1024
    r3 = rindex
    x0 = xindex % 12
    x2 = (xindex // 12288)
    x4 = xindex
    tmp16 = tl.load(in_ptr1 + (r3 + (513*(x1 % 256)) + (131328*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4)) + (525312*x0) + (6303744*x2)), rmask, other=0.0)
    tmp33 = tl.load(in_ptr4 + (r3 + (513*(x1 % 256)) + (131328*((((256*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4)) + (x1 % 256)) // 256) % 4)) + (525312*x2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x1
    tmp1 = tl.full([1], 768, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.broadcast_to(r3, [RBLOCK])
    tmp4 = tl.full([1], 256, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((-197632) + r3 + (257*x1)), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp8 = (tmp7 != 0)
    tmp9 = tl.load(in_ptr1 + (r3 + (513*(x1 % 256)) + (131328*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4)) + (525312*x0) + (6303744*x2)), rmask & tmp6, other=0.0)
    tmp10 = float("-inf")
    tmp11 = tl.where(tmp8, tmp10, tmp9)
    tmp12 = tl.where(tmp6, tmp11, 0.0)
    tmp13 = tl.load(in_ptr1 + (r3 + (513*(x1 % 256)) + (131328*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4)) + (525312*x0) + (6303744*x2)), rmask & tmp2, other=0.0)
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tmp15 = tl.where(tmp2, tmp14, 0.0)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = tl.load(in_ptr2 + ((-393984) + r3 + (513*x1) + (131328*x2)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.where(tmp2, tmp18, 0.0)
    tmp20 = (256*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4)) + (x1 % 256)
    tmp21 = tmp20 < tmp4
    tmp22 = tl.full([1], 257, tl.int64)
    tmp23 = tmp3 < tmp22
    tmp24 = tmp23 & tmp21
    tmp25 = tl.load(in_ptr3 + (r3 + (257*(x1 % 256)) + (65792*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4))), rmask & tmp24, eviction_policy='evict_last', other=0.0)
    tmp26 = (tmp25 != 0)
    tmp27 = tl.load(in_ptr4 + (r3 + (513*(x1 % 256)) + (131328*((((256*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4)) + (x1 % 256)) // 256) % 4)) + (525312*x2)), rmask & tmp24, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.where(tmp26, tmp10, tmp27)
    tmp29 = tl.where(tmp24, tmp28, 0.0)
    tmp30 = tl.load(in_ptr4 + (r3 + (513*(x1 % 256)) + (131328*((((256*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4)) + (x1 % 256)) // 256) % 4)) + (525312*x2)), rmask & tmp21, eviction_policy='evict_last', other=0.0)
    tmp31 = tl.where(tmp23, tmp29, tmp30)
    tmp32 = tl.where(tmp21, tmp31, 0.0)
    tmp34 = tl.where(tmp21, tmp32, tmp33)
    tmp35 = tl.where(tmp2, tmp19, tmp34)
    tmp36 = tmp17 + tmp35
    tmp37 = tl.broadcast_to(tmp36, [RBLOCK])
    tmp39 = tl.where(rmask, tmp37, float("-inf"))
    tmp40 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp39, 0))
    tmp41 = tmp36 - tmp40
    tmp42 = tl.exp(tmp41)
    tmp43 = tl.broadcast_to(tmp42, [RBLOCK])
    tmp45 = tl.where(rmask, tmp43, 0)
    tmp46 = triton_helpers.promote_to_tensor(tl.sum(tmp45, 0))
    tmp47 = tmp42 / tmp46
    tl.store(out_ptr3 + (r3 + (513*x4)), tmp47, rmask)
''')
