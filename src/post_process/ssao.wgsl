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

struct FrameIndex {
    i: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(1) @binding(0)
var t_n: texture_2d<f32>;
@group(1) @binding(1)
var t_d: texture_2d<f32>;
@group(1) @binding(2)
var s: sampler;
@group(1) @binding(3)
var<uniform> frame_index: FrameIndex;

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

fn fast_sqrt(x: f32) -> f32 {
    return f32(0x1FBD1DF5 + (u32(x) >> 1));
}

fn fast_acos(x: f32) -> f32 {
    var res = -0.156583 * abs(x) + PI / 2.0;
    res *= fast_sqrt(1. - abs(x));
    return select(PI - res, res, x >= 0);
}

// https://www.shadertoy.com/view/3tB3z3
fn part1by1(in: u32) -> u32 {
    var x = in;
    x = (x & 0x0000ffffu);
    x = ((x ^ (x << 8u)) & 0x00ff00ffu);
    x = ((x ^ (x << 4u)) & 0x0f0f0f0fu);
    x = ((x ^ (x << 2u)) & 0x33333333u);
    x = ((x ^ (x << 1u)) & 0x55555555u);
    return x;
}

fn compact1by1(in: u32) -> u32 {
    var x = in;
    x = (x & 0x55555555u);
    x = ((x ^ (x >> 1u)) & 0x33333333u);
    x = ((x ^ (x >> 2u)) & 0x0f0f0f0fu);
    x = ((x ^ (x >> 4u)) & 0x00ff00ffu);
    x = ((x ^ (x >> 8u)) & 0x0000ffffu);
    return x;
}

fn pack_morton2x16(v: vec2<u32>) -> u32 {
    return part1by1(v.x) | (part1by1(v.y) << 1);
}

fn unpack_morton2x16(p: u32) -> vec2<u32> {
    return vec2<u32>(compact1by1(p), compact1by1(p >> 1));
}

// https://www.shadertoy.com/view/llGcDm
fn hilbert(in: vec2<i32>, level: i32) -> i32 {
    var p = in;
    var d = 0;
    for (var k = 0; k < level; k++) {
        let n_i = level - k-1;
        let n = u32(n_i);
        let r = vec2<i32>((p.x >> n) & 1, (p.y >> n) & 1);
        d += ((3 * r.x) ^ r.y) << (2 * n);
        if r.y == 0 {
            if r.x == 1 {
                p.x = (i32(1) << n) - i32(1) - p.x;
                p.y = (i32(1) << n) - i32(1) - p.y;
            }
            p = p.yx;
        }
    }
    return d;
}

const g : f32= 1.32471795724474602596;
const a1: f32 = 1.0 / g;
const a2: f32 = 1.0 / (g * g);

// mapping each pixel to a hilbert curve index, then taking a value from the Roberts R2 quasirandom sequence for it
fn hilbert_r2_blue_noisef(p: vec2<u32>) -> vec2<f32> {
    var x = u32(hilbert(vec2<i32>(p), 6)) % (1u << 6u);
    x += 288 * (frame_index.i % 64);
    return vec2<f32>(fract(0.5 + a1 * f32(x)), fract(0.5 + a2 * f32(x)));
}

fn calculate_edges(centerZ: f32, leftZ: f32, rightZ: f32, topZ: f32, bottomZ: f32) -> vec4<f32> {
    var edgesLRTB = vec4<f32>(leftZ, rightZ, topZ, bottomZ) - centerZ;

    let slopeLR = (edgesLRTB.y - edgesLRTB.x) * 0.5;
    let slopeTB = (edgesLRTB.w - edgesLRTB.z) * 0.5;
    let edgesLRTBSlopeAdjusted = edgesLRTB + vec4<f32>(slopeLR, -slopeLR, slopeTB, -slopeTB);
    edgesLRTB = min(abs(edgesLRTB), abs(edgesLRTBSlopeAdjusted));
    return vec4<f32>(saturate((1.25 - edgesLRTB / (centerZ * 0.000011))));
}

// packing/unpacking for edges; 2 bits per edge mean 4 gradient values (0, 0.33, 0.66, 1) for smoother transitions!
fn pack_edges(in: vec4<f32>) -> f32 {
    // integer version:
    // edgesLRTB = saturate(edgesLRTB) * 2.9.xxxx + 0.5.xxxx;
    // return (((uint)edgesLRTB.x) << 6) + (((uint)edgesLRTB.y) << 4) + (((uint)edgesLRTB.z) << 2) + (((uint)edgesLRTB.w));
    //
    // optimized, should be same as above
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
    let noise = hilbert_r2_blue_noisef(vec2<u32>(coords));
    let buffer_size = textureDimensions(t_d);
    let origin = sample_factor * fcoords.xy / vec2<f32>(buffer_size);

    let values_ul = textureGather(0, t_d, s, origin);
    let values_br = textureGather(0, t_d, s, origin, vec2<i32>(1, 1));
    let depth = values_ul.y;

    // viewspace Zs left top right bottom
    let pix_lz = values_ul.x;
    let pix_tz = values_ul.z;
    let pix_rz = values_br.z;
    let pix_bz = values_br.x;

    let edgesLRTB = calculate_edges(depth, pix_lz, pix_rz, pix_tz, pix_bz);
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
    let kernelSize: u32 = 3u;
    let pix_dif = vec2<f32>(1.) / vec2<f32>(buffer_size);
    let camera_distance = sqrt(dot(camera.view_pos.xyz - position, camera.view_pos.xyz - position));
    let wanted_radius = 64. / camera_distance * pix_dif;
    let radius = vec2<f32>(
        min(wanted_radius.x, 64. * pix_dif.x),
        min(wanted_radius.y, 64. * pix_dif.y),
    );
    //let radius = min(min(0.005 / camera_distance, 30. * pix_dif.x), 30. * pix_dif.y);
    for (var i: u32 = 0; i < kernelSize; i += 1) {
        let phi = (noise.x + f32(i)) * PI / f32(kernelSize);
        let dir = vec2<f32>(cos(phi), -sin(phi)) * radius;

        let step_noise_1 = fract(noise.y + f32(i + 1 * 3) * 0.6180339887498948482);
        let step_noise_2 = fract(noise.y + f32(i + 2 * 3) * 0.6180339887498948482);
        let step_noise_3 = fract(noise.y + f32(i + 3 * 3) * 0.6180339887498948482);

        let world_dir = normalize(world_from_screen_coord(origin + dir, depth) - position);
        let ortho_direction_v = world_dir - dot(world_dir, view_dir) * view_dir;
        let slice_plane_normal = normalize(cross(world_dir, view_dir));
        let projected_normal = normal - dot(normal, slice_plane_normal) * slice_plane_normal;
        let sign_n = sign(dot(ortho_direction_v, projected_normal));
        let cos_n = saturate(dot(view_dir, normalize(projected_normal)));
        let n = sign_n * acos(cos_n);

        let sample_coords_1 = vec2<f32>(origin + (0.33 * step_noise_1 + 0.01) * dir);
        let sample_coords_2 = vec2<f32>(origin + (0.33 * step_noise_2 + 0.34) * dir);
        let sample_coords_3 = vec2<f32>(origin + (0.33 * step_noise_3 + 0.67) * dir);
        let sample_coords_1p = vec2<f32>(origin - (0.33 * step_noise_1 + 0.01) * dir);
        let sample_coords_2p = vec2<f32>(origin - (0.33 * step_noise_2 + 0.34) * dir);
        let sample_coords_3p = vec2<f32>(origin - (0.33 * step_noise_3 + 0.67) * dir);
        let sample_depth_1 = textureSample(t_d, s, sample_coords_1).x;
        let sample_depth_2 = textureSample(t_d, s, sample_coords_2).x;
        let sample_depth_3 = textureSample(t_d, s, sample_coords_3).x;
        let sample_depth_1p = textureSample(t_d, s, sample_coords_1p).x;
        let sample_depth_2p = textureSample(t_d, s, sample_coords_2p).x;
        let sample_depth_3p = textureSample(t_d, s, sample_coords_3p).x;

        let sample_1 = world_from_screen_coord(sample_coords_1, sample_depth_1);
        let sample_2 = world_from_screen_coord(sample_coords_2, sample_depth_2);
        let sample_3 = world_from_screen_coord(sample_coords_3, sample_depth_3);
        let sample_1p = world_from_screen_coord(sample_coords_1p, sample_depth_1p);
        let sample_2p = world_from_screen_coord(sample_coords_2p, sample_depth_2p);
        let sample_3p = world_from_screen_coord(sample_coords_3p, sample_depth_3p);

        let d_s_1_squared = dot(sample_1 - position, sample_1 - position);
        let d_s_2_squared = dot(sample_2 - position, sample_2 - position);
        let d_s_3_squared = dot(sample_3 - position, sample_3 - position);
        let d_t_1_squared = dot(sample_1p - position, sample_1p - position);
        let d_t_2_squared = dot(sample_2p - position, sample_2p - position);
        let d_t_3_squared = dot(sample_3p - position, sample_3p - position);

        let d_s_1 = normalize(sample_1 - position);
        let d_s_2 = normalize(sample_2 - position);
        let d_s_3 = normalize(sample_3 - position);
        let d_t_1 = normalize(sample_1p - position);
        let d_t_2 = normalize(sample_2p - position);
        let d_t_3 = normalize(sample_3p - position);

        //let world_radius = 10000.;
        let world_radius = 0.2;
        let l_s_1 = saturate((sqrt(d_s_1_squared) - world_radius) / world_radius);
        let l_s_2 = saturate((sqrt(d_s_2_squared) - world_radius) / world_radius);
        let l_s_3 = saturate((sqrt(d_s_3_squared) - world_radius) / world_radius);
        let l_t_1 = saturate((sqrt(d_t_1_squared) - world_radius) / world_radius);
        let l_t_2 = saturate((sqrt(d_t_2_squared) - world_radius) / world_radius);
        let l_t_3 = saturate((sqrt(d_t_3_squared) - world_radius) / world_radius);
        let dot_s_1 = (1. - l_s_1) * dot(d_s_1, view_dir) + l_s_1 * cos(n + PI * 0.5);
        let dot_s_2 = (1. - l_s_2) * dot(d_s_2, view_dir) + l_s_2 * cos(n + PI * 0.5);
        let dot_s_3 = (1. - l_s_3) * dot(d_s_3, view_dir) + l_s_3 * cos(n + PI * 0.5);
        let dot_t_1 = (1. - l_t_1) * dot(d_t_1, view_dir) + l_t_1 * cos(n - PI * 0.5);
        let dot_t_2 = (1. - l_t_2) * dot(d_t_2, view_dir) + l_t_2 * cos(n - PI * 0.5);
        let dot_t_3 = (1. - l_t_3) * dot(d_t_3, view_dir) + l_t_3 * cos(n - PI * 0.5);

        let cos_h1 = max(max(dot_s_1, dot_s_2), dot_s_3);
        let cos_h2 = max(max(dot_t_1, dot_t_2), dot_t_3);
        let h1 = acos(cos_h1);
        let h2 = -acos(cos_h2);
        let h1p = n + clamp(h1 - n, -PI * 0.5, PI * 0.5);
        let h2p = n + clamp(h2 - n, -PI * 0.5, PI * 0.5);

        let projected_normal_length = sqrt(dot(projected_normal, projected_normal));
        let projected_normal_length_2 = 0.05 * projected_normal_length + (1. - 0.05);

        let local_visibility = 0.25 * projected_normal_length * (- cos(2. * h1p - n) + 2. * cos_n + 2. * (h1p + h2p) * sin(n) - cos(2. * h2p - n));
        visibility += local_visibility / f32(kernelSize);
    }
    out.ssao = visibility;
    return out;
}
