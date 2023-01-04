use wgpu::util::DeviceExt;

use super::data::ColorUniform;
use super::mesh_shader;
use super::DataUniform;
use super::Mesh;
use super::MeshData;
use super::Transform;
use crate::texture;
use crate::util::create_render_pipeline;

pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    pub barycentric_coords: [f32; 3],
    pub distance: f32,
}

impl Vertex for ModelVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
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
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 11]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 14]>() as wgpu::BufferAddress,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

pub struct MeshRenderer {
    vertex_buffer: wgpu::Buffer,
    num_elements: u32,
    transform_bind_group_layout: wgpu::BindGroupLayout,
    transform_bind_group: wgpu::BindGroup,
    transform_buffer: wgpu::Buffer,
    data_uniform: Option<DataUniform>,
    render_pipeline: wgpu::RenderPipeline,
}

pub trait VertexBufferBuilder {
    fn build_vertex_buffer(&self, device: &wgpu::Device, mesh: &Mesh) -> wgpu::Buffer;
}

impl<T> VertexBufferBuilder for Option<&T>
where
    T: VertexBufferBuilder,
{
    fn build_vertex_buffer(&self, device: &wgpu::Device, mesh: &Mesh) -> wgpu::Buffer {
        match self {
            Some(builder) => builder.build_vertex_buffer(device, mesh),
            None => {
                let mut gpu_vertices = Vec::with_capacity(3 * mesh.indices.len());
                for i in 0..mesh.indices.len() {
                    gpu_vertices.push(mesh.vertices[mesh.indices[i][0] as usize]);
                    gpu_vertices.push(mesh.vertices[mesh.indices[i][1] as usize]);
                    gpu_vertices.push(mesh.vertices[mesh.indices[i][2] as usize]);
                    gpu_vertices[3 * i].barycentric_coords = [1., 0., 0.];
                    gpu_vertices[3 * i + 1].barycentric_coords = [0., 1., 0.];
                    gpu_vertices[3 * i + 2].barycentric_coords = [0., 0., 1.];
                }
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{:?} Vertex Buffer", mesh.name)),
                    contents: bytemuck::cast_slice(&gpu_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                })
            }
        }
    }
}

fn build_transform_buffer(
    device: &wgpu::Device,
    name: &str,
    transform: &Transform,
) -> wgpu::Buffer {
    let transform_raw = transform.to_raw();
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{:?} Transform Buffer", name)),
        contents: bytemuck::cast_slice(&[transform_raw]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

fn build_render_pipeline(
    device: &wgpu::Device,
    camera_light_bind_group_layout: &wgpu::BindGroupLayout,
    transform_bind_group_layout: &wgpu::BindGroupLayout,
    color_format: wgpu::TextureFormat,
    data_format: Option<&MeshData>,
    data_uniform: Option<&DataUniform>,
    smooth: bool,
    show_edges: bool,
) -> wgpu::RenderPipeline {
    let render_pipeline_layout = match data_uniform {
        Some(uniform) => device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                camera_light_bind_group_layout,
                transform_bind_group_layout,
                &uniform.bind_group_layout,
            ],
            push_constant_ranges: &[],
        }),
        None => device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[camera_light_bind_group_layout, transform_bind_group_layout],
            push_constant_ranges: &[],
        }),
    };
    let shader = wgpu::ShaderModuleDescriptor {
        label: Some("Normal Shader"),
        source: wgpu::ShaderSource::Wgsl(
            mesh_shader::get_shader(data_format, smooth, show_edges).into(),
        ),
    };
    create_render_pipeline(
        device,
        &render_pipeline_layout,
        color_format,
        Some(texture::Texture::DEPTH_FORMAT),
        &[ModelVertex::desc()],
        shader,
        Some("mesh renderer"),
    )
}

impl MeshRenderer {
    pub fn new(
        device: &wgpu::Device,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        mesh: &Mesh,
    ) -> Self {
        let mesh_data = match &mesh.shown_data {
            Some(data) => mesh.datas.get(data),
            None => None,
        };
        let vertex_buffer = mesh_data.build_vertex_buffer(device, mesh);
        let transform_buffer = build_transform_buffer(device, &mesh.name, &mesh.transform);
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

        let num_elements = (mesh.indices.len() * 3) as u32;
        let data_uniform = if let Some(data) = mesh_data {
            data.build_uniform(device)
        } else {
            Some(ColorUniform::build_uniform(&mesh.color, device))
        };
        let render_pipeline = build_render_pipeline(
            device,
            camera_light_bind_group_layout,
            &transform_bind_group_layout,
            color_format,
            mesh_data,
            data_uniform.as_ref(),
            mesh.smooth,
            mesh.show_edges,
        );
        Self {
            vertex_buffer,
            num_elements,
            transform_bind_group_layout,
            transform_bind_group,
            transform_buffer,
            data_uniform,
            render_pipeline,
        }
    }

    //TODO find some way to split better between mesh and renderer
    //create a struct storing events?
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        mesh: &mut Mesh,
    ) -> bool {
        let mut refresh_screen = false;
        if let Some(data) = mesh.data_to_show.clone() {
            mesh.data_to_show = None;
            mesh.shown_data = data;
            let mesh_data = match &mesh.shown_data {
                Some(data) => mesh.datas.get(data),
                None => None,
            };
            self.vertex_buffer = mesh_data.build_vertex_buffer(device, mesh);
            mesh.property_changed = true;
            refresh_screen = true;
        }
        if mesh.transform_changed {
            let transform_raw = mesh.transform.to_raw();
            queue.write_buffer(
                &self.transform_buffer,
                0,
                bytemuck::cast_slice(&[transform_raw]),
            );
            mesh.transform_changed = false;
            refresh_screen = true;
        }
        if mesh.property_changed {
            let mesh_data = match &mesh.shown_data {
                Some(data) => mesh.datas.get(data),
                None => None,
            };
            let data_uniform = if let Some(data) = mesh_data {
                data.build_uniform(device)
            } else {
                Some(ColorUniform::build_uniform(&mesh.color, device))
            };
            self.data_uniform = data_uniform;
            self.render_pipeline = build_render_pipeline(
                device,
                camera_light_bind_group_layout,
                &self.transform_bind_group_layout,
                color_format,
                mesh_data,
                self.data_uniform.as_ref(),
                mesh.smooth,
                mesh.show_edges,
            );
            mesh.property_changed = false;
            mesh.uniform_changed = false;
            refresh_screen = true;
        } else if mesh.uniform_changed {
            let mesh_data = match &mesh.shown_data {
                Some(data) => mesh.datas.get(data),
                None => None,
            };
            if let Some(mesh_data) = mesh_data {
                mesh_data.refresh_buffer(queue, self.data_uniform.as_ref());
            } else {
                ColorUniform::refresh_buffer(&mesh.color, queue, self.data_uniform.as_ref())
            }
            mesh.uniform_changed = false;
        }
        refresh_screen
    }

    pub fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        render_pass.set_bind_group(1, &self.transform_bind_group, &[]);
        if let Some(data_uniform) = &self.data_uniform {
            render_pass.set_bind_group(2, &data_uniform.bind_group, &[]);
        }
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..self.num_elements, 0..1);
    }
}
