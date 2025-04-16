use super::{SurfaceGeometry, SurfaceSettings};
use crate::camera::Camera;
use crate::data::TransformSettings;
use crate::updater::{ElementPicker, Render};
use crate::util::create_picker_pipeline;
use crate::util::Vertex;
use crate::{texture, SurfaceIndices};
use cgmath::InnerSpace;
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
    @location(1) face_index: u32,
};

// The output we send to our fragment shader
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) face_index: u32,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    // We define the output we want to send over to frag shader
    var out: VertexOutput;
    let model_matrix = transform.model;

    out.face_index = counter.count + model.face_index;

    // We set the \"position\" by using the `clip_position` property
    // We multiply it by the camera position matrix and the instance position matrix
    out.clip_position = camera.view_proj * model_matrix * vec4<f32>(model.position, 1.0);
    return out;
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
//u32 {
    //return bitcast<vec4<f32>>(res);
    // webgl dosen't support rendering to u32, so we have to resort to this
    let res = in.face_index;
    let f1 = f32((res >> u32(24))) / 255.;
    let f2 = f32(((res << u32(8)) >> u32(24))) / 255.;
    let f3 = f32(((res << u32(16)) >> u32(24))) / 255.;
    let f4 = f32(((res << u32(24)) >> u32(24))) / 255.;
    return vec4<f32>(f4, f3, f2, f1);
    //return unpack4x8unorm(in.face_index);
}
";

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct VertexData {
    pub position: [f32; 3],
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
    let mut count = 0;
    for indices in surface.indices.into_iter() {
        for j in 1..indices.len() - 1 {
            let index0 = indices[0];
            let index1 = indices[j];
            let index2 = indices[j + 1];
            let v_indices = [index0, index1, index2];
            for k in 0..3 {
                gpu_vertices.push(VertexData {
                    position: surface.vertices[v_indices[k] as usize],
                    face_index: count,
                });
            }
            count += 1;
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
        //let tot_elements = (surface.vertices.len() + surface.indices.size()) as u32;
        let tot_elements = surface.indices.tot_triangles() as u32;
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

    fn get_element(
        &self,
        surface: &Self::Geometry,
        transform: &TransformSettings,
        camera: &Camera,
        item: u32,
        pos_x: f32,
        pos_y: f32,
    ) -> u32 {
        let indices = &surface.indices;
        let vertices = &surface.vertices;
        let (face_index, face_indices) = match indices {
            SurfaceIndices::Triangles(t) => (item, t[item as usize]),
            SurfaceIndices::Quads(t) => (
                item / 2,
                if item % 2 == 0 {
                    [
                        t[item as usize / 2][0],
                        t[item as usize / 2][1],
                        t[item as usize / 2][2],
                    ]
                } else {
                    [
                        t[item as usize / 2][0],
                        t[item as usize / 2][2],
                        t[item as usize / 2][3],
                    ]
                },
            ),
            SurfaceIndices::Polygons(indices, s) => {
                let mut elapsed = 0;
                let mut index = 0;
                let mut face = [0, 0, 0];
                for (i, size) in s.iter().enumerate() {
                    if elapsed + size - 2 > item {
                        for j in 0..(size - 2) {
                            if elapsed + j == item {
                                index = i as u32;
                                face = [
                                    indices[elapsed as usize + i * 2 + 0],
                                    indices[elapsed as usize + i * 2 + j as usize + 1],
                                    indices[elapsed as usize + i * 2 + j as usize + 2],
                                ];
                                break;
                            }
                        }
                        break;
                    } else {
                        elapsed += size - 2;
                    }
                }
                (index, face)
            }
        };
        let v1: cgmath::Point3<f32> = vertices[face_indices[0] as usize].into();
        let v2: cgmath::Point3<f32> = vertices[face_indices[1] as usize].into();
        let v3: cgmath::Point3<f32> = vertices[face_indices[2] as usize].into();
        let v1 = v1.to_homogeneous();
        let v2 = v2.to_homogeneous();
        let v3 = v3.to_homogeneous();
        let model: cgmath::Matrix4<f32> = transform.to_raw().get_model().into();
        let camera = camera.build_view_projection_matrix();
        let v1 = camera * model * v1;
        let v2 = camera * model * v2;
        let v3 = camera * model * v3;
        let w1 = v1.w;
        let w2 = v2.w;
        let w3 = v3.w;
        let v1 = v1 / v1.w;
        let v2 = v2 / v2.w;
        let v3 = v3 / v3.w;
        let p = cgmath::vec3(pos_x, pos_y, 0.);
        let v1 = cgmath::vec3(v1.x, v1.y, 0.);
        let v2 = cgmath::vec3(v2.x, v2.y, 0.);
        let v3 = cgmath::vec3(v3.x, v3.y, 0.);

        let c3 = (v1 - p).cross(v2 - p).magnitude();
        let c1 = (v2 - p).cross(v3 - p).magnitude();
        let c2 = (v3 - p).cross(v1 - p).magnitude();
        let tot = c3 + c1 + c2;
        let c1 = c1 / tot;
        let c2 = c2 / tot;
        let c3 = c3 / tot;
        let c1 = c1 / w1 / (c1 / w1 + c2 / w2 + c3 / w3);
        let c2 = c2 / w2 / (c1 / w1 + c2 / w2 + c3 / w3);
        let c3 = c3 / w3 / (c1 / w1 + c2 / w2 + c3 / w3);
        if c1 > 0.7 {
            face_indices[0]
        } else if c2 > 0.7 {
            face_indices[1]
        } else if c3 > 0.7 {
            face_indices[2]
        } else {
            vertices.len() as u32 + face_index
        }
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
