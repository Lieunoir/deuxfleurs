use super::{SurfaceGeometry, SurfaceSettings};
use crate::data::TransformSettings;
use crate::texture;
use crate::updater::{ElementPicker, Render};
use crate::util::create_picker_pipeline;
use wgpu::util::DeviceExt;

const PICKER_SHADER: &str = "
// Vertex shader

// Define any uniforms we expect from app
struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}

struct TransformUniform {
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
}

struct CounterUniform {
    count: u32,
    _padding_1: u32,
    _padding_2: u32,
    _padding_3: u32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> light: Light;

@group(1) @binding(0)
var<uniform> counter: CounterUniform;

@group(2) @binding(0)
var<uniform> transform: TransformUniform;

// This is the input from the vertex buffer we created
// We get the properties from our Vertex struct here
// Note the index on location -- this relates to the properties placement in the buffer stride
// e.g. 0 = 1st \"set\" of data, 1 = 2nd \"set\"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) coords: vec3<f32>,
    @location(2) vertex_index_1: u32,
    @location(3) vertex_index_2: u32,
    @location(4) vertex_index_3: u32,
    @location(5) face_index: u32,
};

// The output we send to our fragment shader
struct VertexOutput {
    // This property is \"builtin\" (aka used to render our vertex shader)
    @builtin(position) clip_position: vec4<f32>,
    // These are \"custom\" properties we can create to pass down
    // In this case, we pass the color down
    @location(0) coords: vec3<f32>,
    @location(1) vertex_index_1: u32,
    @location(2) vertex_index_2: u32,
    @location(3) vertex_index_3: u32,
    @location(4) face_index: u32,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    // We define the output we want to send over to frag shader
    var out: VertexOutput;
    let model_matrix = transform.model;

    out.coords = model.coords;
    out.vertex_index_1 = model.vertex_index_1;
    out.vertex_index_2 = model.vertex_index_2;
    out.vertex_index_3 = model.vertex_index_3;
    out.face_index = model.face_index;

    // We set the \"position\" by using the `clip_position` property
    // We multiply it by the camera position matrix and the instance position matrix
    out.clip_position = camera.view_proj * model_matrix * vec4<f32>(model.position, 1.0);
    return out;
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
//u32 {
    // We use the special function `textureSample` to combine the texture data with coords
    let dist = min(1. - in.coords.x, min(1. - in.coords.y, 1. - in.coords.z));
    let thresh = min(0.3, dist);
    let res =   counter.count +
                in.face_index * u32(1. - step(dist, 0.3)) +
                in.vertex_index_1 * u32(step(1. - in.coords.x, thresh)) +
                in.vertex_index_2 * u32(step(1. - in.coords.y, thresh)) +
                in.vertex_index_3 * u32(step(1. - in.coords.z, thresh));
    // webgl dosen't support rendering to u32, so we have to resort to this
    let f1 = f32((res >> u32(24))) / 255.;
    let f2 = f32(((res << u32(8)) >> u32(24))) / 255.;
    let f3 = f32(((res << u32(16)) >> u32(24))) / 255.;
    let f4 = f32(((res << u32(24)) >> u32(24))) / 255.;
    return vec4<f32>(f4, f3, f2, f1);
    //return bitcast<vec4<f32>>(res);
}
";

trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct VertexData {
    pub position: [f32; 3],
    pub barycentric_coords: [f32; 3],
    pub vertex_index_1: u32,
    pub vertex_index_2: u32,
    pub vertex_index_3: u32,
    pub face_index: u32,
}

impl Vertex for VertexData {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 7]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 9]>() as wgpu::BufferAddress,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}

pub struct Picker {
    vertex_buffer: wgpu::Buffer,
    num_elements: u32,
    transform_bind_group: wgpu::BindGroup,
    transform_buffer: wgpu::Buffer,
    render_pipeline: wgpu::RenderPipeline,
    tot_elements: u32,
}

fn build_render_pipeline(
    device: &wgpu::Device,
    camera_light_bind_group_layout: &wgpu::BindGroupLayout,
    counter_bind_group_layout: &wgpu::BindGroupLayout,
    transform_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("SurfaceGeometry Picker Pipeline Layout"),
        bind_group_layouts: &[
            camera_light_bind_group_layout,
            counter_bind_group_layout,
            transform_bind_group_layout,
        ],
        push_constant_ranges: &[],
    });
    let shader = wgpu::ShaderModuleDescriptor {
        label: Some("SurfaceGeometry Picker Shader"),
        source: wgpu::ShaderSource::Wgsl(PICKER_SHADER.into()),
    };
    create_picker_pipeline(
        device,
        &render_pipeline_layout,
        texture::Texture::PICKER_FORMAT,
        Some(texture::Texture::DEPTH_FORMAT),
        &[VertexData::desc()],
        shader,
        Some("surface picker"),
    )
}

fn build_vertex_buffer(device: &wgpu::Device, surface: &SurfaceGeometry) -> wgpu::Buffer {
    let mut gpu_vertices = Vec::with_capacity(3 * surface.internal_indices.len());
    let face_offset = surface.vertices.len();
    //let edges_offset = face_offset + surface.indices.len();
    for (i, indices) in surface.indices.into_iter().enumerate() {
        let bars = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
        //for (bar, index) in bars.iter().zip(indices) {
        for j in 1..indices.len() - 1 {
            let index0 = indices[0];
            let index1 = indices[j];
            let index2 = indices[j + 1];
            let v_indices = [index0, index1, index2];
            for k in 0..3 {
                gpu_vertices.push(VertexData {
                    position: surface.vertices[v_indices[k] as usize],
                    barycentric_coords: bars[k],
                    vertex_index_1: index0,
                    vertex_index_2: index1,
                    vertex_index_3: index2,
                    face_index: (i + face_offset) as u32,
                });
            }
        }
    }
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Picker Vertex Buffer"),
        contents: bytemuck::cast_slice(&gpu_vertices),
        usage: wgpu::BufferUsages::VERTEX,
    })
}

fn build_transform_buffer(device: &wgpu::Device, transform: &TransformSettings) -> wgpu::Buffer {
    let transform_raw = transform.to_raw();
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Picker Transform Buffer"),
        contents: bytemuck::cast_slice(&[transform_raw]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

impl ElementPicker for Picker {
    type Geometry = SurfaceGeometry;
    type Settings = SurfaceSettings;
    fn new(
        surface: &Self::Geometry,
        _settings: &Self::Settings,
        transform: &TransformSettings,
        device: &wgpu::Device,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        counter_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let vertex_buffer = build_vertex_buffer(device, surface);
        let transform_buffer = build_transform_buffer(device, transform);
        let transform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("transform_bind_group_layout"),
            });

        let transform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &transform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: transform_buffer.as_entire_binding(),
            }],
            label: Some("transform_bind_group"),
        });

        let num_elements = (surface.internal_indices.len() * 3) as u32;
        let render_pipeline = build_render_pipeline(
            device,
            camera_light_bind_group_layout,
            counter_bind_group_layout,
            &transform_bind_group_layout,
        );
        let tot_elements = (surface.vertices.len() + surface.indices.size()) as u32;
        Self {
            vertex_buffer,
            num_elements,
            transform_bind_group,
            transform_buffer,
            render_pipeline,
            tot_elements,
        }
    }

    fn update_transform(&self, queue: &mut wgpu::Queue, transform: &TransformSettings) {
        let transform_raw = transform.to_raw();
        queue.write_buffer(
            &self.transform_buffer,
            0,
            bytemuck::cast_slice(&[transform_raw]),
        );
    }

    fn update_settings(&self, _queue: &mut wgpu::Queue, _settings: &Self::Settings) {}

    fn get_total_elements(&self) -> u32 {
        self.tot_elements
    }
}

impl Render for Picker {
    fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        render_pass.set_bind_group(2, &self.transform_bind_group, &[]);
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..self.num_elements, 0..1);
    }
}
