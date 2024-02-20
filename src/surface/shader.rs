use super::SurfaceData;

// macro rules cuz used in format
// forces {{ and }} instead of regular { and }
// kinda ugly, maybe should use some kind of templating
macro_rules! SHADER { () => {"
// Vertex shader

// Define any uniforms we expect from app
struct CameraUniform {{
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}}

struct Light {{
    position: vec3<f32>,
    color: vec3<f32>,
}}

struct TransformUniform {{
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
}}

struct SettingsUniform {{
    color: vec3<f32>,
}}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> light: Light;

@group(1) @binding(0)
var<uniform> transform: TransformUniform;

@group(2) @binding(0)
var<uniform> settings: SettingsUniform;

{}

// This is the input from the vertex buffer we created
// We get the properties from our Vertex struct here
// Note the index on location -- this relates to the properties placement in the buffer stride
// e.g. 0 = 1st \"set\" of data, 1 = 2nd \"set\"
struct VertexInput {{
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) face_normal: vec3<f32>,
    @location(3) barycentric_coords: vec3<f32>,
}};

{}

// The output we send to our fragment shader
struct VertexOutput {{
    // This property is \"builtin\" (aka used to render our vertex shader)
    @builtin(position) clip_position: vec4<f32>,
    // These are \"custom\" properties we can create to pass down
    // In this case, we pass the color down
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) barycentric_coords: vec3<f32>,
    {}
    //@location(2) color: vec3<f32>,
    //@location(3) tex_coords: vec2<f32>,
    //@location(5) distance: f32,
}};

@vertex
fn vs_main(
    model: VertexInput,
    {}
) -> VertexOutput {{
    let model_matrix = transform.model;
    let normal_matrix = transform.normal;

    // We define the output we want to send over to frag shader
    var out: VertexOutput;

    // smooth normals
    {}
    let world_position: vec4<f32> = model_matrix * vec4<f32>(model.position, 1.0);
    out.world_position = world_position.xyz;

    // output assignement
    {}

    //out.tex_coords = model.tex_coords;
    out.barycentric_coords = model.barycentric_coords;
    //out.distance = model.distance;

    // We set the \"position\" by using the `clip_position` property
    // We multiply it by the camera position matrix and the instance position matrix
    out.clip_position = camera.view_proj * model_matrix * vec4<f32>(model.position, 1.0);
    return out;
}}

const PI: f32 = 3.14159;

// PBR functions taken from https://learnopengl.com/PBR/Theory
fn DistributionGGX(N: vec3<f32>, H: vec3<f32>, a: f32) -> f32 {{
    let a2     = a*a;
    let NdotH  = max(dot(N, H), 0.0);
    let NdotH2 = NdotH*NdotH;

    let nom    = a2;
    var denom  = (NdotH2 * (a2 - 1.0) + 1.0);
    denom        = PI * denom * denom;

    return nom / denom;
}}

fn GeometrySchlickGGX(NdotV: f32, k: f32) -> f32
{{
    let nom   = NdotV;
    let denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}}

fn GeometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, k: f32) -> f32
{{
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let ggx1 = GeometrySchlickGGX(NdotV, k);
    let ggx2 = GeometrySchlickGGX(NdotL, k);

    return ggx1 * ggx2;
}}

fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32>
{{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {{
    // We use the special function `textureSample` to combine the texture data with coords

    // We don't need (or want) much ambient light, so 0.1 is fine
    let ambient_strength = 0.1;
    let ambient_color = light.color * ambient_strength;

    let light_dir = normalize(light.position - in.world_position);
    let view_dir = normalize(camera.view_pos.xyz - in.world_position);
    let half_dir = normalize(view_dir + light_dir);

    // smooth shading
    var normal = in.world_normal;
    if(dot(normal, view_dir) < 0.) {{
        normal *= -1.;
    }}

    //var data_color = in.color;
    var data_color = settings.color;
    // use checkerboard or not
    {}
    // show edges
    {}

    let F0 = vec3<f32>(0.04, 0.04, 0.04);
	let D = DistributionGGX(normal, half_dir, 0.6);
    let F = fresnelSchlick(dot(half_dir, normal), F0);
    let G = GeometrySmith(normal, view_dir, light_dir, 0.6);
    let f_ct = D * F * G / (4. * dot(view_dir, normal) * dot(light_dir, normal));
    let kd = 1.0;
    let lambertian = data_color;
    let result = (kd * lambertian + PI * f_ct) * light.color * max(dot(normal, light_dir), 0.0);

    return vec4<f32>(result, 1.);
}}"};}

const COLORMAP_ISOLINES_UNIFORM: &str = "
//vec4 because of alignment issues
struct DataUniform {
    isoline_number: vec4<f32>,
    k_red_vec4: vec4<f32>,
    k_red_vec2: vec4<f32>,
    k_green_vec4: vec4<f32>,
    k_green_vec2: vec4<f32>,
    k_blue_vec4: vec4<f32>,
    k_blue_vec2: vec4<f32>,
}

@group(3) @binding(0)
var<uniform> data_uniform: DataUniform;

//https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7
fn colormap(x: f32) -> vec3<f32> {
    let v4: vec4<f32> = vec4<f32>(1.0, x, x*x, x*x*x);
    let v2: vec4<f32> = v4 * v4.w * x;
    //let v2: vec2<f32> = vec2<f32>(0., 0.);
    return vec3<f32>(
        dot(v4, data_uniform.k_red_vec4) + dot(v2, data_uniform.k_red_vec2),
        dot(v4, data_uniform.k_green_vec4) + dot(v2, data_uniform.k_green_vec2),
        dot(v4, data_uniform.k_blue_vec4) + dot(v2, data_uniform.k_blue_vec2),
    );
}
";

const COLORMAP_ISOLINES: &str = "
    data_color = colormap(in.data);
    let scaled_distance = in.data * data_uniform.isoline_number.x;
    //var testVal = 1.;
    //var modVal = modf(scaled_distance, &testVal);
    var modVal = modf(scaled_distance).fract;
    if(modVal < 0.) {{
        modVal += 1.;
    }}
    let d_dist_x = dpdx(in.data) * data_uniform.isoline_number.x;
    let d_dist_y = dpdy(in.data) * data_uniform.isoline_number.x;
    let d_dist = sqrt(d_dist_x * d_dist_x + d_dist_y * d_dist_y);
    //let remap_1 = smoothstep(0.45, 0.48, modVal);
    //let remap_2 = 1. - smoothstep(0.52, 0.55, modVal);
    let remap_1 = smoothstep(0.5 - d_dist, 0.5, modVal);
    let remap_2 = 1. - smoothstep(0.5, 0.5 + d_dist, modVal);
    data_color = mix(data_color, data_color * .4, min(remap_1, remap_2));
";

const COLORMAP_UNIFORM: &str = "
struct DataUniform {
    k_red_vec4: vec4<f32>,
    k_red_vec2: vec4<f32>,
    k_green_vec4: vec4<f32>,
    k_green_vec2: vec4<f32>,
    k_blue_vec4: vec4<f32>,
    k_blue_vec2: vec4<f32>,
}

@group(3) @binding(0)
var<uniform> data_uniform: DataUniform;

//https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7
fn colormap(x: f32) -> vec3<f32> {
    let v4: vec4<f32> = vec4<f32>(1.0, x, x*x, x*x*x);
    let v2: vec4<f32> = v4 * v4.w * x;
    //let v2: vec2<f32> = vec2<f32>(0., 0.);
    return vec3<f32>(
        dot(v4, data_uniform.k_red_vec4) + dot(v2, data_uniform.k_red_vec2),
        dot(v4, data_uniform.k_green_vec4) + dot(v2, data_uniform.k_green_vec2),
        dot(v4, data_uniform.k_blue_vec4) + dot(v2, data_uniform.k_blue_vec2),
    );
}
";

const COLORMAP: &str = "
    data_color = colormap(in.data);
";

const COLOR_UNIFORM: &str = "
struct DataUniform {
    color: vec4<f32>,
}

@group(3) @binding(0)
var<uniform> data_uniform: DataUniform;
";

const CHECKERBOARD_UNIFORM: &str = "
struct DataUniform {
    color_1: vec4<f32>,
    color_2: vec4<f32>,
    period: f32,
}

@group(3) @binding(0)
var<uniform> data_uniform: DataUniform;
";

const CHECKERBOARD: &str = "
    let check_period = data_uniform.period;
    //Trash value
    //var check_mod_period = 1.;
    let tex_color = data_uniform.color_1.xyz;
    //let check_mod_x = modf(in.data.x * check_period, &check_mod_period);
    let check_mod_x = modf(in.data.x * check_period).fract;
    let check_min_x = 2. * (max(abs(check_mod_x), 1. - abs(check_mod_x)) - .5);
    let check_mod_y = modf(in.data.y * check_period).fract;
    let check_min_y = 2. * (max(abs(check_mod_y), 1. - abs(check_mod_y)) - .5);

    let v_check = (check_min_x - .5) * (check_min_y - .5);

    // Not exactly the derivative of v, sign is wrong due to abs/max stuff but it doesn't matter coz it all ends up squared
    let d_check_x = 2. * check_period * ((check_min_y - 0.5) * dpdx(in.data.x) + (check_min_x - 0.5) * dpdx(in.data.y));
    let d_check_y = 2. * check_period * ((check_min_x - 0.5) * dpdy(in.data.y) + (check_min_y - 0.5) * dpdy(in.data.x));
    let d_check = sqrt(d_check_x * d_check_x + d_check_y * d_check_y);
    let s_check = smoothstep(-d_check, d_check, v_check);
    data_color = mix(tex_color, data_uniform.color_2.xyz, s_check);
";

/*
const FLAT_NORMAL_INTERPOLATION: &str = "
    let tan_x = dpdx(in.world_position);
    let tan_y = dpdy(in.world_position);
    var normal = normalize(cross(tan_x, tan_y));
";*/
const FLAT_NORMAL_INTERPOLATION: &str = "
    out.world_normal = normalize((normal_matrix * vec4<f32>(model.face_normal, 0.0)).xyz);
";
const SMOOTH_NORMAL_INTERPOLATION: &str = "
    out.world_normal = normalize((normal_matrix * vec4<f32>(model.normal, 0.0)).xyz);
";

const WITH_EDGE_SHADER: &str = "
    let d_bary_x = dpdx(in.barycentric_coords);
    let d_bary_y = dpdy(in.barycentric_coords);
    let d_bary = sqrt(d_bary_x * d_bary_x + d_bary_y * d_bary_y);
    //let thickness = .5;
    //let falloff = 1.;
    //let remap = smoothstep(d_bary * thickness, d_bary * (thickness + falloff), in.barycentric_coords);
    let thickness = 1.5;
    let remap = smoothstep(vec3<f32>(0.), d_bary * thickness, in.barycentric_coords);
    let wire_frame = min(remap.x, min(remap.y, remap.z));
    let edge_color = vec3<f32>(0.02, 0.02, 0.02);

    data_color = mix(data_color, edge_color, 1. - wire_frame);
";

pub fn get_shader(data_format: Option<&SurfaceData>, smooth: bool, show_edge: bool) -> String {
    let uniform = if let Some(mesh_data) = data_format {
        match mesh_data {
            SurfaceData::UVMap(_, _) | SurfaceData::UVCornerMap(_, _) => CHECKERBOARD_UNIFORM,
            SurfaceData::VertexScalar(_, _) => COLORMAP_ISOLINES_UNIFORM,
            SurfaceData::FaceScalar(_, _) => COLORMAP_UNIFORM,
            _ => "",
        }
    } else {
        COLOR_UNIFORM
    };

    let (data_decl, data_out) = match data_format {
        Some(data) => match data {
            SurfaceData::UVMap(..) | SurfaceData::UVCornerMap(..) => ("
struct DataInput {
    @location(4) data: vec2<f32>,
};", "@location(3) data: vec2<f32>,"),
            SurfaceData::VertexScalar(..) | SurfaceData::FaceScalar(..) => ("
struct DataInput {
    @location(4) data: f32,
};", "@location(3) data: f32,"),
            SurfaceData::Color(..) => ("
struct DataInput {
    @location(4) data: vec3<f32>,
};", "@location(3) data: vec3<f32>,"),
        },
        None => ("", ""),
    };

    let (data_input, data_assign) = match data_format {
        Some(data) => ("
    data: DataInput,", "out.data = data.data;"),
        None => ("", ""),
    };

    let normal_interpolation = if smooth {
        SMOOTH_NORMAL_INTERPOLATION
    } else {
        FLAT_NORMAL_INTERPOLATION
    };
    let render_modif = match data_format {
        Some(SurfaceData::UVMap(_, _)) | Some(SurfaceData::UVCornerMap(_, _)) => CHECKERBOARD,
        Some(SurfaceData::VertexScalar(_, _)) => COLORMAP_ISOLINES,
        Some(SurfaceData::FaceScalar(_, _)) => COLORMAP,
        _ => "",
    };
    let edge_shader = if show_edge { WITH_EDGE_SHADER } else { "" };
    format!(
        SHADER!(),
        uniform, data_decl, data_out, data_input, normal_interpolation, data_assign, render_modif, edge_shader
    )
}
