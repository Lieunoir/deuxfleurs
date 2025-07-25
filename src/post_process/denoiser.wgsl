@group(0) @binding(0)
var source_ao: texture_2d<f32>;
@group(0) @binding(1)
var source_edges: texture_2d<f32>;
@group(0) @binding(2)
var s: sampler;

const pos = array(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));

@vertex
fn vs_main(
    @builtin(vertex_index) index: u32,
) -> @builtin(position) vec4<f32> {
    return vec4<f32>(pos[index], 0.0, 1.0);
}

fn unpack_edges(packedValIn: f32) -> vec4<f32> {
    let packedVal = u32(packedValIn * 255.5);
    var edgesLRTB: vec4<f32>;
    edgesLRTB.x = f32((packedVal >> 6) & 0x03) / 3.0;          // there's really no need for mask (as it's an 8 bit input) but I'll leave it in so it doesn't cause any trouble in the future
    edgesLRTB.y = f32((packedVal >> 4) & 0x03) / 3.0;
    edgesLRTB.z = f32((packedVal >> 2) & 0x03) / 3.0;
    edgesLRTB.w = f32((packedVal >> 0) & 0x03) / 3.0;

    return saturate(edgesLRTB);
}

fn add_sample(ssaoValue: f32, edgeValue: f32, sum: ptr<function, f32>, sumWeight: ptr<function, f32>) {
    let weight = edgeValue;

    *sum += (weight * ssaoValue);
    *sumWeight += weight;
}

@fragment
fn fs_main(@builtin(position) fcoords: vec4<f32>) -> @location(0) f32 {
    let buffer_size = textureDimensions(source_ao);
    let gather_offset = - vec2<f32>(0.25) / vec2<f32>(buffer_size);
    let gatherCenter = fcoords.xy / vec2<f32>(buffer_size) + gather_offset;

    // let blurAmount = 1.2f;
    let blurAmount = 1.2f;
    let diagWeight = 0.65 * 0.5;

    // gather edge and visibility quads, used later
    let edgesQ0 = textureGather(0, source_edges, s, gatherCenter, vec2<i32>(0, 0));
    let edgesQ1 = textureGather(0, source_edges, s, gatherCenter, vec2<i32>(2, 0));
    let edgesQ2 = textureGather(0, source_edges, s, gatherCenter, vec2<i32>(1, 2));

    let visQ0 = textureGather(0, source_ao, s, gatherCenter, vec2<i32>(0, 0));
    let visQ1 = textureGather(0, source_ao, s, gatherCenter, vec2<i32>(2, 0));
    let visQ2 = textureGather(0, source_ao, s, gatherCenter, vec2<i32>(0, 2));
    let visQ3 = textureGather(0, source_ao, s, gatherCenter, vec2<i32>(2, 2));

    let side = 0;

    let edgesL_LRTB = unpack_edges(select(edgesQ0.y, edgesQ0.x, side == 0));
    let edgesT_LRTB = unpack_edges(select(edgesQ1.w, edgesQ0.z, side == 0));
    let edgesR_LRTB = unpack_edges(select(edgesQ1.y, edgesQ0.x, side == 0));
    let edgesB_LRTB = unpack_edges(select(edgesQ2.z, edgesQ0.w, side == 0));

    var edgesC_LRTB = unpack_edges(select(edgesQ1.x, edgesQ0.y, side == 0));

        // Edges aren't perfectly symmetrical: edge detection algorithm does not guarantee that a left edge on the right pixel will match the right edge on the left pixel (although
        // they will match in majority of cases). This line further enforces the symmetricity, creating a slightly sharper blur. Works real nice with TAA.
    edgesC_LRTB *= vec4<f32>(edgesL_LRTB.y, edgesR_LRTB.x, edgesT_LRTB.w, edgesB_LRTB.z);

    let leak_threshold = 2.5;
    let leak_strength = 0.5;
    let edginess = (saturate(4.0 - leak_threshold - dot(edgesC_LRTB, vec4<f32>(1.))) / (4 - leak_threshold)) * leak_strength;
    edgesC_LRTB = saturate(edgesC_LRTB + edginess);

        // for diagonals; used by first and second pass
    let weightTL = diagWeight * (edgesC_LRTB.x * edgesL_LRTB.z + edgesC_LRTB.z * edgesT_LRTB.x);
    let weightTR = diagWeight * (edgesC_LRTB.z * edgesT_LRTB.y + edgesC_LRTB.y * edgesR_LRTB.z);
    let weightBL = diagWeight * (edgesC_LRTB.w * edgesB_LRTB.x + edgesC_LRTB.x * edgesL_LRTB.w);
    let weightBR = diagWeight * (edgesC_LRTB.y * edgesR_LRTB.w + edgesC_LRTB.w * edgesB_LRTB.y);

        // first pass
    let ssaoValue = select(visQ1.x, visQ0.y, (side == 0));
    let ssaoValueL = select(visQ0.y, visQ0.x, (side == 0));
    let ssaoValueT = select(visQ1.w, visQ0.z, (side == 0));
    let ssaoValueR = select(visQ1.y, visQ1.x, (side == 0));
    let ssaoValueB = select(visQ3.w, visQ2.z, (side == 0));
    let ssaoValueTL = select(visQ0.z, visQ0.w, (side == 0));
    let ssaoValueBR = select(visQ3.z, visQ3.w, (side == 0));
    let ssaoValueTR = select(visQ1.z, visQ1.w, (side == 0));
    let ssaoValueBL = select(visQ2.z, visQ2.w, (side == 0));

    var sumWeight = blurAmount;
    var sum = ssaoValue * sumWeight;

    add_sample(ssaoValueL, edgesC_LRTB.x, &sum, &sumWeight);
    add_sample(ssaoValueR, edgesC_LRTB.y, &sum, &sumWeight);
    add_sample(ssaoValueT, edgesC_LRTB.z, &sum, &sumWeight);
    add_sample(ssaoValueB, edgesC_LRTB.w, &sum, &sumWeight);

    add_sample(ssaoValueTL, weightTL, &sum, &sumWeight);
    add_sample(ssaoValueTR, weightTR, &sum, &sumWeight);
    add_sample(ssaoValueBL, weightBL, &sum, &sumWeight);
    add_sample(ssaoValueBR, weightBR, &sum, &sumWeight);

    return(sum / sumWeight);
}
