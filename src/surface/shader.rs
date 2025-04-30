use super::SurfaceData;

// macro rules cuz used in format
// forces {{ and }} instead of regular { and }
// kinda ugly, maybe should use some kind of templating
macro_rules! SHADER {
    () => {
        "
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

struct Jitter {{
    jitter: vec4<f32>,
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
@group(0) @binding(2)
var<uniform> jitter: Jitter;

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
    @location(1) normal: vec4<f32>,
    @location(2) face_normal: vec4<f32>,
    //@location(3) barycentric_coords: vec3<f32>,
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
    let b_codes = u32(model.normal.w * 127.);
    //let b_codes = 4;
    //let b_codes = max(-1, min(1, model.normal.w));
    let b_1 = select(vec3<f32>(0.), vec3<f32>(1., 0., 0.), bool(b_codes & 4));
    let b_2 = select(vec3<f32>(0.), vec3<f32>(0., 1., 0.), bool(b_codes & 2));
    let b_3 = select(vec3<f32>(0.), vec3<f32>(0., 0., 1.), bool(b_codes & 1));
    out.barycentric_coords = b_1 + b_2 + b_3;
    //out.distance = model.distance;

    // We set the \"position\" by using the `clip_position` property
    // We multiply it by the camera position matrix and the instance position matrix
    let clip_pos = camera.view_proj * model_matrix * vec4<f32>(model.position, 1.0);
    out.clip_position = clip_pos + jitter.jitter * clip_pos.w;
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

struct MaterialOutput {{
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
}};

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> MaterialOutput {{
    // We use the special function `textureSample` to combine the texture data with coords
    let view_dir = normalize(camera.view_pos.xyz - in.world_position);

    let normal = select(in.world_normal, -in.world_normal, dot(in.world_normal, view_dir) < 0.);

    //var data_color = in.color;
    var data_color = settings.color;
    // use checkerboard or not
    {}
    // show edges
    {}

    var out: MaterialOutput;
    out.albedo = vec4<f32>(data_color, 0.6);
    //out.normal = vec4<f32>((normal + vec3<f32>(256. / 255.)) * 255. / 256. / 2., 0.);
    out.normal = vec4<f32>((normal + vec3<f32>(1.)) / 2. , 0.);
    //out.normal = vec4<f32>(normal, 0.);

    return out;
}}"
    };
}

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
    min: f32,
    max: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(3) @binding(0)
var<uniform> data_uniform: DataUniform;

fn linear_from_gamma(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb < vec3<f32>(0.04045);
    let lower = srgb / vec3<f32>(12.92);
    let higher = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return select(higher, lower, cutoff);
}

//https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7
fn colormap(dist: f32) -> vec3<f32> {
    let x = (clamp(dist, data_uniform.min, data_uniform.max) - data_uniform.min) / (data_uniform.max - data_uniform.min);
    let v4: vec4<f32> = vec4<f32>(1.0, x, x*x, x*x*x);
    let v2: vec4<f32> = v4 * v4.w * x;
    //let v2: vec2<f32> = vec2<f32>(0., 0.);
    let res = vec3<f32>(
        dot(v4, data_uniform.k_red_vec4) + dot(v2, data_uniform.k_red_vec2),
        dot(v4, data_uniform.k_green_vec4) + dot(v2, data_uniform.k_green_vec2),
        dot(v4, data_uniform.k_blue_vec4) + dot(v2, data_uniform.k_blue_vec2),
    );
    return linear_from_gamma(res);
}
";

const COLORMAP_ISOLINES: &str = "
    data_color = colormap(in.data);
    let dist = (clamp(in.data, data_uniform.min, data_uniform.max) - data_uniform.min) / (data_uniform.max - data_uniform.min);
    let scaled_distance = dist * data_uniform.isoline_number.x;
    //var testVal = 1.;
    //var modVal = modf(scaled_distance, &testVal);
    var modVal = modf(scaled_distance).fract;
    if(modVal < 0.) {{
        modVal += 1.;
    }}
    let d_dist_x = dpdx(dist) * data_uniform.isoline_number.x;
    let d_dist_y = dpdy(dist) * data_uniform.isoline_number.x;
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
    min: f32,
    max: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(3) @binding(0)
var<uniform> data_uniform: DataUniform;

fn linear_from_gamma(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb < vec3<f32>(0.04045);
    let lower = srgb / vec3<f32>(12.92);
    let higher = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return select(higher, lower, cutoff);
}

//https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7
fn colormap(dist: f32) -> vec3<f32> {
    let x = (clamp(dist, data_uniform.min, data_uniform.max) - data_uniform.min) / (data_uniform.max - data_uniform.min);
    let v4: vec4<f32> = vec4<f32>(1.0, x, x*x, x*x*x);
    let v2: vec4<f32> = v4 * v4.w * x;
    //let v2: vec2<f32> = vec2<f32>(0., 0.);
    let res = vec3<f32>(
        dot(v4, data_uniform.k_red_vec4) + dot(v2, data_uniform.k_red_vec2),
        dot(v4, data_uniform.k_green_vec4) + dot(v2, data_uniform.k_green_vec2),
        dot(v4, data_uniform.k_blue_vec4) + dot(v2, data_uniform.k_blue_vec2),
    );
    return linear_from_gamma(res);
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
    let tex_color = data_uniform.color_1.xyz;
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
    out.world_normal = normalize((normal_matrix * vec4<f32>(model.face_normal.xyz, 0.0)).xyz);
";
const SMOOTH_NORMAL_INTERPOLATION: &str = "
    out.world_normal = normalize((normal_matrix * vec4<f32>(model.normal.xyz, 0.0)).xyz);
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
            SurfaceData::UVMap(..) | SurfaceData::UVCornerMap(..) => (
                "
struct DataInput {
    @location(4) data: vec2<f32>,
};",
                "@location(3) data: vec2<f32>,",
            ),
            SurfaceData::VertexScalar(..) | SurfaceData::FaceScalar(..) => (
                "
struct DataInput {
    @location(4) data: f32,
};",
                "@location(3) data: f32,",
            ),
            SurfaceData::Color(..) => (
                "
struct DataInput {
    @location(4) data: vec3<f32>,
};",
                "@location(3) data: vec3<f32>,",
            ),
        },
        None => ("", ""),
    };

    let (data_input, data_assign) = match data_format {
        Some(_data) => (
            "
    data: DataInput,",
            "out.data = data.data;",
        ),
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
        uniform,
        data_decl,
        data_out,
        data_input,
        normal_interpolation,
        data_assign,
        render_modif,
        edge_shader
    )
}

pub const SHADOW_SHADER: &str = "
struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    min_bb: vec2<f32>,
    max_bb: vec2<f32>,
    shadow_proj: mat4x4<f32>,
}

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}

struct Jitter {
    jitter: vec4<f32>,
}

struct TransformUniform {
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
}


@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> light: Light;
@group(0) @binding(2)
var<uniform> jitter: Jitter;

@group(1) @binding(0)
var<uniform> transform: TransformUniform;


// This is the input from the vertex buffer we created
// We get the properties from our Vertex struct here
// Note the index on location -- this relates to the properties placement in the buffer stride
// e.g. 0 = 1st \"set\" of data, 1 = 2nd \"set\"
struct VertexInput {
    @builtin(vertex_index) index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) face_normal: vec4<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> @builtin(position) vec4<f32> {
    let clip_pos = camera.shadow_proj * transform.model * vec4<f32>(model.position, 1.0);
    return clip_pos + jitter.jitter * clip_pos.w;
    //return clip_pos;
}

// Fragment shader
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) f32 {
    return 1.;
}";
