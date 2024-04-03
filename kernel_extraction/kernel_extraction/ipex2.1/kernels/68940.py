

# Original file: ./doctr_det_predictor___60.0/doctr_det_predictor___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/o2/co2ny64ylutdyun72j5hsj4c2uz3kpqfs2vzxkpj6g7pmo4kepnq.py
# Source Nodes: [add, add_1, l__self___fpn_upsample, l__self___fpn_upsample_1], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.mul, aten.rsub, aten.sub]
# add => add_135
# add_1 => add_141
# l__self___fpn_upsample => add_130, add_134, convert_element_type_114, convert_element_type_116, iota, mul_171, mul_173, mul_179, mul_180, sub_59, sub_60
# l__self___fpn_upsample_1 => _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, add_136, add_138, add_139, add_140, convert_element_type_120, convert_element_type_122, iota_2, mul_181, mul_183, mul_185, mul_186, mul_187, mul_188, mul_189, mul_190, sub_61, sub_62, sub_63, sub_64
triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_3 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[256, 16384], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_3(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 128)
    x1 = xindex % 128
    y0 = yindex
    x3 = xindex
    tmp84 = tl.load(in_ptr3 + (y0 + (256*x3)), ymask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.49606299212598426
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.int32)
    tmp9 = x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp11 + tmp4
    tmp13 = tmp12 * tmp6
    tmp14 = tmp13.to(tl.int32)
    tmp15 = tl.load(in_ptr0 + (tmp14 + (64*tmp8) + (4096*y0)), ymask)
    tmp16 = tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 * tmp2
    tmp19 = tmp18 + tmp4
    tmp20 = 0.49206349206349204
    tmp21 = tmp19 * tmp20
    tmp22 = tmp21.to(tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 - tmp23
    tmp25 = tmp2 - tmp24
    tmp26 = tmp15 * tmp25
    tmp27 = tl.load(in_ptr1 + (tmp14 + (64*tmp8) + (4096*y0)), ymask)
    tmp28 = tmp27 * tmp24
    tmp29 = tmp26 + tmp28
    tmp30 = tl.load(in_ptr2 + (y0 + (256*tmp14) + (16384*tmp8)), ymask)
    tmp31 = tmp29 + tmp30
    tmp32 = libdevice.ceil(tmp7)
    tmp33 = 63.0
    tmp34 = triton_helpers.minimum(tmp32, tmp33)
    tmp35 = tmp34.to(tl.int32)
    tmp36 = tl.load(in_ptr0 + (tmp14 + (64*tmp35) + (4096*y0)), ymask)
    tmp37 = tmp36 * tmp25
    tmp38 = tl.load(in_ptr1 + (tmp14 + (64*tmp35) + (4096*y0)), ymask)
    tmp39 = tmp38 * tmp24
    tmp40 = tmp37 + tmp39
    tmp41 = tl.load(in_ptr2 + (y0 + (256*tmp14) + (16384*tmp35)), ymask)
    tmp42 = tmp40 + tmp41
    tmp43 = tmp8.to(tl.float32)
    tmp44 = tmp7 - tmp43
    tmp45 = tmp2 - tmp44
    tmp46 = tmp31 * tmp45
    tmp47 = tmp42 * tmp44
    tmp48 = tmp46 + tmp47
    tmp49 = tmp14.to(tl.float32)
    tmp50 = tmp13 - tmp49
    tmp51 = tmp2 - tmp50
    tmp52 = tmp48 * tmp51
    tmp53 = libdevice.ceil(tmp13)
    tmp54 = triton_helpers.minimum(tmp53, tmp33)
    tmp55 = tmp54.to(tl.int32)
    tmp56 = tl.load(in_ptr0 + (tmp55 + (64*tmp8) + (4096*y0)), ymask)
    tmp57 = tmp55
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tmp58 * tmp2
    tmp60 = tmp59 + tmp4
    tmp61 = tmp60 * tmp20
    tmp62 = tmp61.to(tl.int32)
    tmp63 = tmp62.to(tl.float32)
    tmp64 = tmp61 - tmp63
    tmp65 = tmp2 - tmp64
    tmp66 = tmp56 * tmp65
    tmp67 = tl.load(in_ptr1 + (tmp55 + (64*tmp8) + (4096*y0)), ymask)
    tmp68 = tmp67 * tmp64
    tmp69 = tmp66 + tmp68
    tmp70 = tl.load(in_ptr2 + (y0 + (256*tmp55) + (16384*tmp8)), ymask)
    tmp71 = tmp69 + tmp70
    tmp72 = tl.load(in_ptr0 + (tmp55 + (64*tmp35) + (4096*y0)), ymask)
    tmp73 = tmp72 * tmp65
    tmp74 = tl.load(in_ptr1 + (tmp55 + (64*tmp35) + (4096*y0)), ymask)
    tmp75 = tmp74 * tmp64
    tmp76 = tmp73 + tmp75
    tmp77 = tl.load(in_ptr2 + (y0 + (256*tmp55) + (16384*tmp35)), ymask)
    tmp78 = tmp76 + tmp77
    tmp79 = tmp71 * tmp45
    tmp80 = tmp78 * tmp44
    tmp81 = tmp79 + tmp80
    tmp82 = tmp81 * tmp50
    tmp83 = tmp52 + tmp82
    tmp85 = tmp83 + tmp84
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + (16384*y0)), tmp52, ymask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3 + (16384*y0)), tmp82, ymask)
    tl.store(out_ptr2 + (y0 + (256*x3)), tmp85, ymask)
''')
