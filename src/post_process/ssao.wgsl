struct Parameters {
    frame_offset: u32,
    slices: u32,
    samples: u32,
    _pad3: u32,
    depth_linearize_mul: f32,
    depth_linearize_add: f32,
    _pad4: u32,
    _pad5: u32,
}

@group(0) @binding(0)
var t_n: texture_2d<f32>;
@group(0) @binding(1)
var t_d: texture_2d<f32>;
@group(0) @binding(2)
var s: sampler;
@group(0) @binding(3)
var<uniform> param: Parameters;
@group(0) @binding(4)
var hilbert: texture_2d<u32>;

const pos = array(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));

@vertex
fn vs_main(
    @builtin(vertex_index) index: u32,
) -> @builtin(position) vec4<f32> {
    return vec4<f32>(pos[index], 0.0, 1.0);
}

const PI: f32 = 3.14159265359;
const tan_pi_0125 = 0.41421356237;

fn view_from_screen_coord(coord: vec2<f32>, linear_depth_sample: f32) -> vec3<f32> {
    // reconstruct view-space position from the screen coordinate and view space depth.
    return vec3<f32>(
        - (vec2<f32>(2. * 0.90225565, -2.) * coord + vec2<f32>(-1. * 0.90225565, 1.)) * linear_depth_sample * tan_pi_0125,
        linear_depth_sample
    );
}

fn linearize_depth(depth: f32) -> f32 {
    return param.depth_linearize_mul / (param.depth_linearize_add + depth);
}

fn fast_sqrt(x: f32) -> f32 {
    return bitcast<f32>(0x1FBD1DF5 + (bitcast<i32>(x) >> 1));
}

fn fast_acos(x: f32) -> f32 {
    let abs_x = abs(x);
    var res = 1.5707963267948966 - 0.1565827644218014 * abs_x;
    res *= fast_sqrt(1.0 - abs_x);
    return select(PI - res, res, x >= 0);
}

const g : f32= 1.32471795724474602596;
const a1: f32 = 1.0 / g;
const a2: f32 = 1.0 / (g * g);

// mapping each pixel to a hilbert curve index, then taking a value from the Roberts R2 quasirandom sequence for it
fn hilbert_r2_blue_noisef(p: vec2<u32>) -> vec2<f32> {
    var x = textureLoad(hilbert, vec2<u32>(p.x % 64, p.y % 64), 0).x;
    x += param.frame_offset;
    return vec2<f32>(fract(0.5 + vec2<f32>(a1, a2) * f32(x)));
}

fn calculate_edges(centerZ: f32, leftZ: f32, rightZ: f32, topZ: f32, bottomZ: f32) -> vec4<f32> {
    var edgesLRTB = vec4<f32>(leftZ, rightZ, topZ, bottomZ) - centerZ;

    let slopeLR = (edgesLRTB.y - edgesLRTB.x) * 0.5;
    let slopeTB = (edgesLRTB.w - edgesLRTB.z) * 0.5;
    let edgesLRTBSlopeAdjusted = edgesLRTB + vec4<f32>(slopeLR, -slopeLR, slopeTB, -slopeTB);
    edgesLRTB = min(abs(edgesLRTB), abs(edgesLRTBSlopeAdjusted));
    //remember: centerZ < 0.
    return vec4<f32>(saturate((1.25 + edgesLRTB / (centerZ * 0.011))));
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
const kernel_size: u32 = 2u;
const samples_per_slice = 2u;
const phi_mul = PI / f32(kernel_size);
const vis_scale = 0.25 / f32(kernel_size);

@fragment
fn fs_main(@builtin(position) fcoords: vec4<f32>) -> FragmentOutput {
    var out: FragmentOutput;
    let buffer_size = vec2<f32>(textureDimensions(t_d));
    let pix_dif = vec2<f32>(1.) / buffer_size;
    let coords = vec2<u32>(floor(fcoords.xy * sample_factor));
    let noise = hilbert_r2_blue_noisef(coords);
    let origin = sample_factor * fcoords.xy * pix_dif;
    let gather_offset = - vec2<f32>(0.25) * pix_dif;
    let values_ul = textureGather(0, t_d, s, origin + gather_offset);
    let values_br = textureGather(0, t_d, s, origin + gather_offset, vec2<i32>(1, 1));
    let depth = values_ul.y;

    // viewspace Zs left top right bottom
    let pix_lz = values_ul.x;
    let pix_tz = values_ul.z;
    let pix_rz = values_br.z;
    let pix_bz = values_br.x;

    let edgesLRTB = calculate_edges(
        depth,
        pix_lz,
        pix_rz,
        pix_tz,
        pix_bz
    );
    let packed_edges = pack_edges(edgesLRTB);
    out.edges = packed_edges;

    let position = view_from_screen_coord(origin, depth);
    let normal_sample = textureLoad(t_n, coords, 0).xyz;
    let normal = normalize(normal_sample * 2. - vec3<f32>(1.));
    let view_dir = normalize(-position);

    //if abs(normal_sample[0]) + abs(normal_sample[1]) + abs(normal_sample[2]) < 0.01 {
	//    discard;
    //}

    // GTAO
    var visibility = 0.0;
    //let kernel_size: u32 = param.slices;
    //let samples_per_slice = param.samples;
    let world_distance = linearize_depth(1.);
    let world_radius_mul = 10. / world_distance;
    //let world_radius = 0.02 * world_distance;
    let wanted_screen_radius = 2. * world_distance * pix_dif / depth;
    let radius = vec2<f32>(
        64. * min(wanted_screen_radius.x, pix_dif.x),
        64. * min(wanted_screen_radius.y, pix_dif.y),
    );
    //let radius = min(min(0.005 / camera_distance, 30. * pix_dif.x), 30. * pix_dif.y);
    for (var i: u32 = 0; i < kernel_size; i += 1) {
        let phi = (noise.x + f32(i)) * phi_mul;
        let cos_phi = cos(phi);
        let sin_phi = sin(phi);
        //let sin_phi = fast_sqrt(1. - cos_phi * cos_phi);
        let dir = vec2<f32>(cos_phi, -sin_phi) * radius;
        let world_dir = vec3<f32>(cos_phi, sin_phi, 0.);
        let slice_plane_normal = normalize(cross(view_dir, world_dir));
        let projected_normal = normal - dot(normal, slice_plane_normal) * slice_plane_normal;
        let projected_normal_len = length(projected_normal);
        let ortho_direction_v = cross(projected_normal, slice_plane_normal);
        let sign_n = sign(dot(view_dir, ortho_direction_v));
        let cos_n = dot(view_dir, projected_normal) / projected_normal_len;
        let n = sign_n * fast_acos(cos_n);
        let sin_n = sign_n * fast_sqrt(1. - cos_n * cos_n);
        let cos_n_plus_pi_2 = -sin_n;
        let cos_n_minus_pi_2 = sin_n;
        var cos_h1 = cos_n_plus_pi_2;
        var cos_h2 = cos_n_minus_pi_2;

        for (var j: u32 = 0; j < samples_per_slice; j += 1) {
            let step_noise = fract(noise.y + f32(i + j * samples_per_slice) * 0.6180339887498948482);
            let sample_offset = pow((step_noise + f32(j)) / f32(samples_per_slice), 2.) * dir;

            let sample_offset_length = length(sample_offset * buffer_size);
            let mip_level = clamp(log2(sample_offset_length) - 3.3, 0., 4.);

            let sample_coords_1 = vec2<f32>(origin + sample_offset);
            let sample_coords_2 = vec2<f32>(origin - sample_offset);
            let sample_depth_1 = textureSampleLevel(t_d, s, sample_coords_1, mip_level).x;
            let sample_depth_2 = textureSampleLevel(t_d, s, sample_coords_2, mip_level).x;

            let sample_1 = view_from_screen_coord(sample_coords_1, sample_depth_1);
            let sample_2 = view_from_screen_coord(sample_coords_2, sample_depth_2);
            let dir_1 = sample_1 - position;
            let dir_2 = sample_2 - position;

            let d_s_1_norm = length(dir_1);
            let d_s_2_norm = length(dir_2);

            let d_s_1 = dir_1 / d_s_1_norm;
            let d_s_2 = dir_2 / d_s_2_norm;

            // (d - 0.02 * world_distance) * 5. / (0.02 * world_distance)
            // = (d / (0.02 * world_distance) - 1.) * 5.)
            //let l_s_1 = saturate((d_s_1_norm - world_radius) * 5. / world_radius);
            //let l_s_2 = saturate((d_s_2_norm - world_radius) * 5. / world_radius);
            let l_s_1 = saturate(d_s_1_norm * world_radius_mul - 5.);
            let l_s_2 = saturate(d_s_2_norm * world_radius_mul - 5.);
            let dot_s_1 = mix(dot(d_s_1, view_dir), cos_n_plus_pi_2, l_s_1);
            let dot_s_2 = mix(dot(d_s_2, view_dir), cos_n_minus_pi_2, l_s_2);
            cos_h1 = max(cos_h1, dot_s_1);
            cos_h2 = max(cos_h2, dot_s_2);
        }

        let h1p = fast_acos(cos_h1);
        let h2p = -fast_acos(cos_h2);
        //let h1p = n + max(h1 - n, -PI * 0.5);
        //let h2p = n + min(h2 - n, PI * 0.5);
        let vis_1 = - cos(2. * h1p - n) + 2. * h1p * sin_n;
        let vis_2 = - cos(2. * h2p - n) + 2. * h2p * sin_n;
        let local_visibility = projected_normal_len * (vis_1 + vis_2 + 2. * cos_n);
        visibility += local_visibility;
    }

    var scaled_visibility = pow(0.25 * visibility / f32(kernel_size), 1.35);

    /*
    if abs(dpdx(depth)) < 0.0000002 {
        scaled_visibility -= dpdx(scaled_visibility) * (f32(coords.x & 1) - 0.5);
    }
    if abs(dpdy(depth)) < 0.0000002 {
        scaled_visibility -= dpdy(scaled_visibility) * (f32(coords.y & 1) - 0.5);
    }*/
    out.ssao = scaled_visibility;
    return out;
}
