pub const COLORMAP_UNIFORM: &str = "
struct ColorMapUniform {
    k_red_vec1: vec4<f32>,
    k_red_vec2: vec4<f32>,
    k_green_vec1: vec4<f32>,
    k_green_vec2: vec4<f32>,
    k_blue_vec1: vec4<f32>,
    k_blue_vec2: vec4<f32>,
    min: f32,
    max: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(3) @binding(0)
var<uniform> colormap_uniform: ColorMapUniform;

fn linear_from_gamma(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb < vec3<f32>(0.04045);
    let lower = srgb / vec3<f32>(12.92);
    let higher = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return select(higher, lower, cutoff);
}

//https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7
fn colormap(dist: f32) -> vec3<f32> {
    let x = (clamp(dist, colormap_uniform.min, colormap_uniform.max) - colormap_uniform.min) / (colormap_uniform.max - colormap_uniform.min);
    let v4: vec4<f32> = vec4<f32>(1.0, x, x*x, x*x*x);
    let v2: vec4<f32> = v4 * v4.w * x;
    //let v2: vec2<f32> = vec2<f32>(0., 0.);
    let res = vec3<f32>(
        dot(v4, colormap_uniform.k_red_vec1) + dot(v2, colormap_uniform.k_red_vec2),
        dot(v4, colormap_uniform.k_green_vec1) + dot(v2, colormap_uniform.k_green_vec2),
        dot(v4, colormap_uniform.k_blue_vec1) + dot(v2, colormap_uniform.k_blue_vec2),
    );
    return linear_from_gamma(res);
}
";
