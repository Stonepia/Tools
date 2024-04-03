

# Original file: ./tnt_s_patch16_224___60.0/tnt_s_patch16_224___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/is/cis3anwchjj4ekv3dp6dyt4o6ssn2d5rwxbs65ngozi5enq4x235.py
# Source Nodes: [add_8, l__self___blocks_1_mlp_in_drop2, l__self___blocks_1_norm1_proj, l__self___blocks_1_proj, l__self___blocks_2_attn_in_qk, l__self___blocks_2_norm_in], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.native_layer_norm]
# add_8 => add_31
# l__self___blocks_1_mlp_in_drop2 => clone_26
# l__self___blocks_1_norm1_proj => add_32, clone_27, mul_30, rsqrt_9, sub_12, var_mean_9
# l__self___blocks_1_proj => convert_element_type_59
# l__self___blocks_2_attn_in_qk => convert_element_type_77
# l__self___blocks_2_norm_in => add_43, mul_41
triton_poi_fused__to_copy_add_clone_native_layer_norm_25 = async_compile.triton('triton_poi_fused__to_copy_add_clone_native_layer_norm_25', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: '*fp16', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_native_layer_norm_25', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_clone_native_layer_norm_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9633792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    x1 = (xindex // 384)
    x4 = (xindex // 24)
    x3 = xindex % 24
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + ((16*x1) + (x0 // 24)), None)
    tmp6 = tl.load(in_ptr3 + ((16*x1) + (x0 // 24)), None)
    tmp13 = tl.load(in_ptr4 + (x0 % 24), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0 % 24), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr6 + (x3), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x3), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp3 - tmp4
    tmp7 = 24.0
    tmp8 = tmp6 / tmp7
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp3 - tmp18
    tmp21 = tmp20 / tmp7
    tmp22 = tmp21 + tmp9
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp19 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp28.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp17, None)
    tl.store(out_ptr1 + (x2), tmp29, None)
''')
