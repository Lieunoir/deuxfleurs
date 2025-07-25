struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
    view_inv: mat4x4<f32>,
}

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> light: Light;

struct Parameters {
    frame_index: u32,
    slices: u32,
    samples: u32,
    _pad3: u32,
    depth_linearize_mul: f32,
    depth_linearize_add: f32,
    _pad4: u32,
    _pad5: u32,
}

@group(1) @binding(0)
var t_n: texture_2d<f32>;
@group(1) @binding(1)
var t_d: texture_2d<f32>;
@group(1) @binding(2)
var s: sampler;
@group(1) @binding(3)
var<uniform> param: Parameters;
@group(1) @binding(4)
var hilbert: texture_2d<u32>;

const pos = array(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));

@vertex
fn vs_main(
    @builtin(vertex_index) index: u32,
) -> @builtin(position) vec4<f32> {
    return vec4<f32>(pos[index], 0.0, 1.0);
}

const PI: f32 = 3.14159265359;

fn world_from_screen_coord(coord: vec2<f32>, depth_sample: f32) -> vec3<f32> {
    // reconstruct world-space position from the screen coordinate.
    let posClip = vec4(coord.x * 2.0 - 1.0, 1.0 - 2.0 * coord.y, depth_sample, 1.0);
    let posWorldW = camera.view_inv * posClip;
    let posWorld = posWorldW.xyz / posWorldW.www;
    return posWorld;
}

fn linearize_depth(depth: f32) -> f32 {
    return param.depth_linearize_mul / (param.depth_linearize_add - depth);
}

fn fast_sqrt(x: f32) -> f32 {
    return bitcast<f32>(0x1FBD1DF5 + (bitcast<i32>(x) >> 1));
}

fn fast_acos(x: f32) -> f32 {
    var res = -0.156583 * abs(x) + PI / 2.0;
    res *= fast_sqrt(1. - abs(x));
    return select(PI - res, res, x >= 0);
}

const g : f32= 1.32471795724474602596;
const a1: f32 = 1.0 / g;
const a2: f32 = 1.0 / (g * g);

// mapping each pixel to a hilbert curve index, then taking a value from the Roberts R2 quasirandom sequence for it
fn hilbert_r2_blue_noisef(p: vec2<i32>) -> vec2<f32> {
    var x = textureLoad(hilbert, vec2<i32>(p.x % 64, p.y % 64), 0).x;
    x += 288 * (param.frame_index % 64);
    return vec2<f32>(fract(0.5 + a1 * f32(x)), fract(0.5 + a2 * f32(x)));
}

fn calculate_edges(centerZ: f32, leftZ: f32, rightZ: f32, topZ: f32, bottomZ: f32) -> vec4<f32> {
    var edgesLRTB = vec4<f32>(leftZ, rightZ, topZ, bottomZ) - centerZ;

    let slopeLR = (edgesLRTB.y - edgesLRTB.x) * 0.5;
    let slopeTB = (edgesLRTB.w - edgesLRTB.z) * 0.5;
    let edgesLRTBSlopeAdjusted = edgesLRTB + vec4<f32>(slopeLR, -slopeLR, slopeTB, -slopeTB);
    edgesLRTB = min(abs(edgesLRTB), abs(edgesLRTBSlopeAdjusted));
    return vec4<f32>(saturate((1.25 - edgesLRTB / (centerZ * 0.0011))));
}

// packing/unpacking for edges; 2 bits per edge mean 4 gradient values (0, 0.33, 0.66, 1) for smoother transitions!
fn pack_edges(in: vec4<f32>) -> f32 {
    var edgesLRTB = round(saturate(in) * 2.9);
    return dot(edgesLRTB, vec4<f32>(64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0)) ;
}

const sample_factor: f32 = 1.;

struct FragmentOutput {
    @location(0) ssao: f32,
    @location(1) edges: f32,
}

@fragment
fn fs_main(@builtin(position) fcoords: vec4<f32>) -> FragmentOutput {
    var out: FragmentOutput;
    let coords = vec2<i32>(floor(fcoords.xy * sample_factor));
    let noise = hilbert_r2_blue_noisef(coords);
    let buffer_size = textureDimensions(t_d);
    let origin = sample_factor * fcoords.xy / vec2<f32>(buffer_size);
    let gather_offset = - vec2<f32>(0.25) / vec2<f32>(buffer_size);
    let values_ul = textureGather(0, t_d, s, origin + gather_offset);
    let values_br = textureGather(0, t_d, s, origin + gather_offset, vec2<i32>(1, 1));
    let depth = values_ul.y;

    // viewspace Zs left top right bottom
    let pix_lz = values_ul.x;
    let pix_tz = values_ul.z;
    let pix_rz = values_br.z;
    let pix_bz = values_br.x;

    let edgesLRTB = calculate_edges(
        linearize_depth(depth),
        linearize_depth(pix_lz),
        linearize_depth(pix_rz),
        linearize_depth(pix_tz),
        linearize_depth(pix_bz)
    );
    let packed_edges = pack_edges(edgesLRTB);
    out.edges = packed_edges;

    let position = world_from_screen_coord(origin, depth);
    let normal_sample = textureLoad(t_n, coords, 0).xyz;
    let normal = normalize(normal_sample * 2. - vec3<f32>(1.));
    let view_dir = normalize(camera.view_pos.xyz - position);

    if abs(normal_sample[0]) + abs(normal_sample[1]) + abs(normal_sample[2]) < 0.01 {
	    discard;
    }

    // GTAO
    var visibility = 0.0;
    let kernelSize: u32 = param.slices;
    let pix_dif = vec2<f32>(1.) / vec2<f32>(buffer_size);
    let world_distance = linearize_depth(1.);
    let camera_distance = linearize_depth(depth) / world_distance;
    let wanted_radius = 64. / camera_distance * pix_dif;
    let radius = vec2<f32>(
        min(wanted_radius.x, 32. * pix_dif.x),
        min(wanted_radius.y, 32. * pix_dif.y),
    );
    //let radius = min(min(0.005 / camera_distance, 30. * pix_dif.x), 30. * pix_dif.y);
    for (var i: u32 = 0; i < kernelSize; i += 1) {
        let phi = (noise.x + f32(i)) * PI / f32(kernelSize);
        let dir = vec2<f32>(cos(phi), -sin(phi)) * radius;
        let world_dir = normalize(world_from_screen_coord(origin + dir, depth) - position);
        let ortho_direction_v = world_dir - dot(world_dir, view_dir) * view_dir;
        let slice_plane_normal = normalize(cross(world_dir, view_dir));
        let projected_normal = normal - dot(normal, slice_plane_normal) * slice_plane_normal;
        let sign_n = sign(dot(ortho_direction_v, projected_normal));
        let cos_n = saturate(dot(view_dir, normalize(projected_normal)));
        let n = sign_n * fast_acos(cos_n);
        let sin_n = sin(n);
        let cos_n_plus_pi_2 = -sin_n;
        let cos_n_minus_pi_2 = sin_n;
        var cos_h1 = cos_n_plus_pi_2;
        var cos_h2 = cos_n_minus_pi_2;

        //let world_radius = 10000.;
        let world_radius = 0.02 * world_distance;

        for (var j: u32 = 0; j < param.samples; j += 1) {
            let step_noise = fract(noise.y + f32(i + j * param.samples) * 0.6180339887498948482);

            let sample_coords_1 = vec2<f32>(origin + (step_noise + f32(j)) * dir / f32(param.samples));
            let sample_coords_2 = vec2<f32>(origin - (step_noise + f32(j)) * dir / f32(param.samples));
            let sample_depth_1 = textureSample(t_d, s, sample_coords_1).x;
            let sample_depth_2 = textureSample(t_d, s, sample_coords_2).x;

            let sample_1 = world_from_screen_coord(sample_coords_1, sample_depth_1);
            let sample_2 = world_from_screen_coord(sample_coords_2, sample_depth_2);

            let d_s_1_squared = dot(sample_1 - position, sample_1 - position);
            let d_s_2_squared = dot(sample_2 - position, sample_2 - position);

            let d_s_1 = normalize(sample_1 - position);
            let d_s_2 = normalize(sample_2 - position);

            let l_s_1 = saturate((fast_sqrt(d_s_1_squared) - world_radius) / world_radius);
            let l_s_2 = saturate((fast_sqrt(d_s_2_squared) - world_radius) / world_radius);
            let dot_s_1 = (1. - l_s_1) * dot(d_s_1, view_dir) + l_s_1 * cos_n_plus_pi_2;
            let dot_s_2 = (1. - l_s_2) * dot(d_s_2, view_dir) + l_s_2 * cos_n_minus_pi_2;
            cos_h1 = max(cos_h1, dot_s_1);
            cos_h2 = max(cos_h2, dot_s_2);
        }

        let h1 = fast_acos(cos_h1);
        let h2 = -fast_acos(cos_h2);
        let h1p = n + clamp(h1 - n, -PI * 0.5, PI * 0.5);
        let h2p = n + clamp(h2 - n, -PI * 0.5, PI * 0.5);

        let projected_normal_length = fast_sqrt(dot(projected_normal, projected_normal));
        let local_visibility = 0.25 * projected_normal_length * (- cos(2. * h1p - n) + 2. * cos_n + 2. * (h1p + h2p) * sin_n - cos(2. * h2p - n));
        visibility += local_visibility / f32(kernelSize);
    }
    out.ssao = visibility;
    return out;
}
