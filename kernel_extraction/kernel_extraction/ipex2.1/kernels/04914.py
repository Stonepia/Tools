

# Original file: ./swin_base_patch4_window7_224___60.0/swin_base_patch4_window7_224___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/kd/ckdgovjytak2wd77rvcs54g75lemesbstha6p6juxoqtbttsjbay.py
# Source Nodes: [getattr_getattr_l__mod___layers___1___blocks___1___norm2], Original ATen: [aten.native_layer_norm]
# getattr_getattr_l__mod___layers___1___blocks___1___norm2 => add_34, add_35, convert_element_type_32, convert_element_type_33, mul_31, mul_32, rsqrt_9, sub_13, var_mean_9
triton_red_fused_native_layer_norm_26 = async_compile.triton('triton_red_fused_native_layer_norm_26', '''
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
    size_hints=[65536, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_26', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_native_layer_norm_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    x3 = xindex
    tmp9_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (256*(x0 % 28)) + (7168*((((28*(x0 // 28)) + (x0 % 28)) // 28) % 28)) + (200704*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r2 + (256*((x0 % 28) % 7)) + (1792*(((((28*(x0 // 28)) + (x0 % 28)) // 28) % 28) % 7)) + (12544*((x0 % 28) // 7)) + (50176*(((((28*(x0 // 28)) + (x0 % 28)) // 28) % 28) // 7)) + (200704*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r2 + (256*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r2 + (256*(((25 + (x0 % 28)) % 28) % 7)) + (1792*(((25 + (x0 // 28)) % 28) % 7)) + (12544*(((25 + (x0 % 28)) % 28) // 7)) + (50176*(((25 + (x0 // 28)) % 28) // 7)) + (200704*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp9_mean_next, tmp9_m2_next, tmp9_weight_next = triton_helpers.welford_reduce(
            tmp8, tmp9_mean, tmp9_m2, tmp9_weight,
        )
        tmp9_mean = tl.where(rmask & xmask, tmp9_mean_next, tmp9_mean)
        tmp9_m2 = tl.where(rmask & xmask, tmp9_m2_next, tmp9_m2)
        tmp9_weight = tl.where(rmask & xmask, tmp9_weight_next, tmp9_weight)
    tmp9_tmp, tmp10_tmp, tmp11_tmp = triton_helpers.welford(
        tmp9_mean, tmp9_m2, tmp9_weight, 1
    )
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp12 = tl.load(in_ptr0 + (r2 + (256*(x0 % 28)) + (7168*((((28*(x0 // 28)) + (x0 % 28)) // 28) % 28)) + (200704*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr1 + (r2 + (256*((x0 % 28) % 7)) + (1792*(((((28*(x0 // 28)) + (x0 % 28)) // 28) % 28) % 7)) + (12544*((x0 % 28) // 7)) + (50176*(((((28*(x0 // 28)) + (x0 % 28)) // 28) % 28) // 7)) + (200704*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr2 + (r2 + (256*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr3 + (r2 + (256*(((25 + (x0 % 28)) % 28) % 7)) + (1792*(((25 + (x0 // 28)) % 28) % 7)) + (12544*(((25 + (x0 % 28)) % 28) // 7)) + (50176*(((25 + (x0 // 28)) % 28) // 7)) + (200704*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tmp12 + tmp13
        tmp16 = tmp14 + tmp15
        tmp18 = tmp16 + tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp19 - tmp9
        tmp21 = 256.0
        tmp22 = tmp10 / tmp21
        tmp23 = 1e-05
        tmp24 = tmp22 + tmp23
        tmp25 = libdevice.rsqrt(tmp24)
        tmp26 = tmp20 * tmp25
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp26 * tmp28
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp29 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (256*x3)), tmp33, rmask & xmask)
''')
