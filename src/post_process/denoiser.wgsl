@group(0) @binding(0)
var source_ao: texture_2d<f32>;
@group(0) @binding(1)
var source_edges: texture_2d<f32>;
@group(0) @binding(2)
var depth: texture_2d<f32>;
@group(0) @binding(3)
var s: sampler;

const pos = array(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));

@vertex
fn vs_main(
    @builtin(vertex_index) index: u32,
) -> @builtin(position) vec4<f32> {
    return vec4<f32>(pos[index], 0.0, 1.0);
}

fn gaussian(sigma: f32, x: f32) -> f32 {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

//const OFFSET = array(0.0, 1.3846153846, 3.2307692308);
//const WEIGHT = array(0.2270270270, 0.3162162162, 0.0702702703);
const OFFSET = array(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
//const WEIGHT = array(0.2270270270, 0.1945945946, 0.1216216216,
//    0.0540540541, 0.0162162162);
const sigmaV = 0.02;
const sigmaS = 2.;

@fragment
fn fs_horizontal(@builtin(position) fcoords: vec4<f32>) -> @location(0) f32 {
    let buffer_size = vec2<f32>(textureDimensions(source_ao));
    let coords = fcoords.xy / buffer_size;
    var weight = textureSample(source_ao, s, coords).x;
    let dpx = buffer_size.x;
    let I = textureSampleLevel(depth, s, coords, 1.).r;
    var tot_weight = 1.;
    for (var r = 1; r < 7; r++) {
        let Z0 = textureSampleLevel(depth, s, coords + vec2<f32>(OFFSET[r] / dpx, 0.), 1.0).x;
        let Z1 = textureSampleLevel(depth, s, coords - vec2<f32>(OFFSET[r] / dpx, 0.), 1.0).x;
        let w0 = gaussian(sigmaV, abs(I - Z0));
        let w1 = gaussian(sigmaV, abs(I - Z1));
        tot_weight += (w0 + w1) * gaussian(sigmaS, OFFSET[r]);
        weight += w0 * textureSample(source_ao, s, coords + vec2<f32>(OFFSET[r] / dpx, 0.)).x * gaussian(sigmaS, OFFSET[r]);
        weight += w1 * textureSample(source_ao, s, coords - vec2<f32>(OFFSET[r] / dpx, 0.)).x * gaussian(sigmaS, OFFSET[r]);
    }
    return weight / tot_weight;
}

@fragment
fn fs_vertical(@builtin(position) fcoords: vec4<f32>) -> @location(0) f32 {
    let buffer_size = vec2<f32>(textureDimensions(source_ao));
    let coords = fcoords.xy / buffer_size;
    var weight = textureSample(source_ao, s, coords).x;
    let dpy = buffer_size.y;
    let I = textureSampleLevel(depth, s, coords, 1.).r;
    var tot_weight = 1.;
    for (var r = 1; r < 7; r++) {
        let Z0 = textureSampleLevel(depth, s, coords + vec2<f32>(0., OFFSET[r] / dpy), 1.0).x;
        let Z1 = textureSampleLevel(depth, s, coords - vec2<f32>(0., OFFSET[r] / dpy), 1.0).x;
        let w0 = gaussian(sigmaV, abs(I - Z0));
        let w1 = gaussian(sigmaV, abs(I - Z1));
        tot_weight += (w0 + w1) * gaussian(sigmaS, OFFSET[r]);
        weight += w0 * textureSample(source_ao, s, coords + vec2<f32>(0., OFFSET[r] / dpy)).x * gaussian(sigmaS, OFFSET[r]);
        weight += w1 * textureSample(source_ao, s, coords - vec2<f32>(0., OFFSET[r] / dpy)).x * gaussian(sigmaS, OFFSET[r]);
    }
    return weight / tot_weight;
}
