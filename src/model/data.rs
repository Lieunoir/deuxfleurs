use crate::model::renderer::VertexBufferBuilder;
use crate::model::Mesh;
use crate::ui::UiMeshDataElement;
use wgpu::util::DeviceExt;

pub struct DataUniform {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColorSettings {
    color: [f32; 4],
}

impl DataUniformBuilder for ColorSettings {}

impl ColorSettings {
    pub fn build_uniform(color: &[f32; 4], device: &wgpu::Device) -> DataUniform {
        let uniform = ColorSettings { color: *color };
        uniform.build_uniform(device)
    }

    pub fn refresh_buffer(
        color: &[f32; 4],
        queue: &mut wgpu::Queue,
        data_uniform: Option<&DataUniform>,
    ) {
        let uniform = ColorSettings { color: *color };
        uniform.refresh_buffer(queue, data_uniform.unwrap());
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexScalarSettings {
    pub isoline_number: f32,
    _padding: [f32; 3],
}

impl Default for VertexScalarSettings {
    fn default() -> Self {
        Self {
            isoline_number: 0.,
            _padding: [0.; 3],
        }
    }
}

impl DataUniformBuilder for VertexScalarSettings {}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UVMapSettings {
    pub color_1: [f32; 4],
    pub color_2: [f32; 4],
    pub period: f32,
    _padding: [f32; 3],
}

impl Default for UVMapSettings {
    fn default() -> Self {
        Self {
            color_1: [0.9, 0.9, 0.9, 1.],
            color_2: [0.6, 0.2, 0.4, 1.],
            period: 100.,
            _padding: [0.; 3],
        }
    }
}

impl DataUniformBuilder for UVMapSettings {}

#[non_exhaustive]
pub enum MeshData {
    Color(Vec<[f32; 3]>),
    FaceScalar(Vec<f32>),
    VertexScalar(Vec<f32>, VertexScalarSettings),
    //TODO think about edge ordering
    //EdgeScalar(Vec<f32>),
    UVMap(Vec<[f32; 2]>, UVMapSettings),
    UVCornerMap(Vec<[f32; 2]>, UVMapSettings),
}

impl MeshData {
    pub fn build_uniform(&self, device: &wgpu::Device) -> Option<DataUniform> {
        match self {
            MeshData::VertexScalar(_, uniform) => Some(uniform.build_uniform(device)),
            MeshData::UVMap(_, uniform) => Some(uniform.build_uniform(device)),
            MeshData::UVCornerMap(_, uniform) => Some(uniform.build_uniform(device)),
            // Maybe use empty uniform instead of none?
            _ => None,
        }
    }

    pub fn refresh_buffer(&self, queue: &mut wgpu::Queue, data_uniform: Option<&DataUniform>) {
        match self {
            MeshData::VertexScalar(_, uniform) => {
                uniform.refresh_buffer(queue, data_uniform.unwrap())
            }
            MeshData::UVMap(_, uniform) => uniform.refresh_buffer(queue, data_uniform.unwrap()),
            MeshData::UVCornerMap(_, uniform) => {
                uniform.refresh_buffer(queue, data_uniform.unwrap())
            }
            _ => (),
        }
    }
}

trait DataUniformBuilder: bytemuck::Pod + Copy {
    fn build_uniform(self, device: &wgpu::Device) -> DataUniform {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Data buffer"),
            contents: bytemuck::cast_slice(&[self]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("data_bind_group_layout"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("data_bind_group"),
        });
        DataUniform {
            bind_group_layout,
            bind_group,
            buffer,
        }
    }

    fn refresh_buffer(self, queue: &mut wgpu::Queue, data_uniform: &DataUniform) {
        queue.write_buffer(&data_uniform.buffer, 0, bytemuck::cast_slice(&[self]));
    }
}

impl UiMeshDataElement for MeshData {
    fn draw(&mut self, ui: &mut egui::Ui) -> bool {
        use egui::Widget;
        match self {
            MeshData::UVMap(_, data_uniform) | MeshData::UVCornerMap(_, data_uniform) => {
                let mut changed = false;
                //ui.add(egui::Slider::new(&mut data_uniform.period, 0.0..=100.0).text("Period"));
                changed |= egui::Slider::new(&mut data_uniform.period, 0.0..=100.0)
                    .text("Period")
                    .clamp_to_range(false)
                    .ui(ui)
                    .changed();
                let mut color_1 = egui::Rgba::from_rgba_unmultiplied(
                    data_uniform.color_1[0],
                    data_uniform.color_1[1],
                    data_uniform.color_1[2],
                    data_uniform.color_1[3],
                );
                let mut color_2 = egui::Rgba::from_rgba_unmultiplied(
                    data_uniform.color_2[0],
                    data_uniform.color_2[1],
                    data_uniform.color_2[2],
                    data_uniform.color_2[3],
                );
                ui.horizontal(|ui| {
                    changed |= egui::widgets::color_picker::color_edit_button_rgba(
                        ui,
                        &mut color_1,
                        egui::widgets::color_picker::Alpha::Opaque,
                    )
                    .changed();
                    ui.label("Checkerboard color 1");
                });
                ui.horizontal(|ui| {
                    changed |= egui::widgets::color_picker::color_edit_button_rgba(
                        ui,
                        &mut color_2,
                        egui::widgets::color_picker::Alpha::Opaque,
                    )
                    .changed();
                    ui.label("Checkerboard color 2");
                });
                data_uniform.color_1 = color_1.to_array();
                data_uniform.color_2 = color_2.to_array();
                changed
            }
            MeshData::VertexScalar(_, data_uniform) => {
                egui::Slider::new(&mut data_uniform.isoline_number, 0.0..=100.0)
                    .text("Isolines")
                    .clamp_to_range(false)
                    .ui(ui)
                    .changed()
            }
            _ => false,
        }
    }
}

impl VertexBufferBuilder for MeshData {
    fn build_vertex_buffer(&self, device: &wgpu::Device, mesh: &Mesh) -> wgpu::Buffer {
        let mut gpu_vertices = Vec::with_capacity(3 * mesh.indices.size());
        let mut i = 0;
        for (face_index, face) in mesh.indices.into_iter().enumerate() {
            for j in 1..face.len() - 1 {
                gpu_vertices.push(mesh.internal_vertices[mesh.internal_indices[i][0] as usize]);
                gpu_vertices.push(mesh.internal_vertices[mesh.internal_indices[i][1] as usize]);
                gpu_vertices.push(mesh.internal_vertices[mesh.internal_indices[i][2] as usize]);
                gpu_vertices[3 * i].face_normal = mesh.face_normals[face_index];
                gpu_vertices[3 * i + 1].face_normal = mesh.face_normals[face_index];
                gpu_vertices[3 * i + 2].face_normal = mesh.face_normals[face_index];
                if face.len() == 3 {
                    gpu_vertices[3 * i].barycentric_coords = [1., 0., 0.];
                    gpu_vertices[3 * i + 1].barycentric_coords = [0., 1., 0.];
                    gpu_vertices[3 * i + 2].barycentric_coords = [0., 0., 1.];
                } else if j == 1 {
                    gpu_vertices[3 * i].barycentric_coords = [1., 1., 0.];
                    gpu_vertices[3 * i + 1].barycentric_coords = [0., 1., 0.];
                    gpu_vertices[3 * i + 2].barycentric_coords = [0., 0., 1.];
                } else if j == face.len() - 2 {
                    gpu_vertices[3 * i].barycentric_coords = [1., 0., 1.];
                    gpu_vertices[3 * i + 1].barycentric_coords = [0., 1., 0.];
                    gpu_vertices[3 * i + 2].barycentric_coords = [0., 0., 1.];
                } else {
                    gpu_vertices[3 * i].barycentric_coords = [1., 1., 1.];
                    gpu_vertices[3 * i + 1].barycentric_coords = [0., 1., 0.];
                    gpu_vertices[3 * i + 2].barycentric_coords = [0., 0., 1.];
                }
                i += 1;
            }
        }
        match self {
            MeshData::Color(colors) => {
                for (i, vertex) in gpu_vertices.iter_mut().enumerate() {
                    let color = colors[mesh.internal_indices[i / 3][i % 3] as usize];
                    vertex.color = color;
                }
            }
            MeshData::VertexScalar(datas, _) => {
                let mut min_d = datas[0];
                let mut max_d = datas[0];
                for data in datas {
                    if *data > max_d {
                        max_d = *data;
                    }
                    if *data < min_d {
                        min_d = *data;
                    }
                }
                for (i, vertex) in gpu_vertices.iter_mut().enumerate() {
                    let data = datas[mesh.internal_indices[i / 3][i % 3] as usize];
                    let t = (data - min_d) / (max_d - min_d);
                    vertex.color[0] = t * t;
                    vertex.color[1] = 2. * t * (1. - t);
                    vertex.color[2] = (1. - t) * (1. - t);
                    vertex.distance = t;
                }
            }
            MeshData::FaceScalar(datas) => {
                let mut min_d = datas[0];
                let mut max_d = datas[0];
                for data in datas {
                    if *data > max_d {
                        max_d = *data;
                    }
                    if *data < min_d {
                        min_d = *data;
                    }
                }
                let mut k = 0;
                for (face, data) in mesh.indices.into_iter().zip(datas) {
                    let t = (data - min_d) / (max_d - min_d);
                    let color = [t * t, 2. * t * (1. - t), (1. - t) * (1. - t)];
                    for _i in 1..face.len() - 1 {
                        gpu_vertices[3 * k].color = color;
                        gpu_vertices[3 * k + 1].color = color;
                        gpu_vertices[3 * k + 2].color = color;
                        gpu_vertices[3 * k].distance = t;
                        gpu_vertices[3 * k + 1].distance = t;
                        gpu_vertices[3 * k + 2].distance = t;
                        k += 1;
                    }
                }
            }
            MeshData::UVMap(uv_map, _) => {
                for (i, vertex) in gpu_vertices.iter_mut().enumerate() {
                    let uv = uv_map[mesh.internal_indices[i / 3][i % 3] as usize];
                    vertex.tex_coords = uv;
                }
            }
            MeshData::UVCornerMap(uv_map, _) => {
                //TODO this but for polygonal faces
                for (vertex, uv) in gpu_vertices.iter_mut().zip(uv_map) {
                    vertex.tex_coords = *uv;
                }
            }
            _ => (),
        }
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?} Vertex Buffer", mesh.name)),
            contents: bytemuck::cast_slice(&gpu_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        })
    }
}
