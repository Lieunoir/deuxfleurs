use crate::data::{ColorMap, ColorMapValues, IsolineSettings, UVMapSettings};
use crate::data::{ColorSettings, DataUniform};
use crate::model::renderer::VertexBufferBuilder;
use crate::model::Mesh;
use crate::ui::UiDataElement;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VertexScalarSettingsBuffer {
    isoline: IsolineSettings,
    colormap: ColorMapValues,
}

#[derive(Copy, Clone)]
pub struct VertexScalarSettings {
    pub isoline: IsolineSettings,
    pub colormap: ColorMap,
}

#[derive(Copy, Clone)]
pub struct FaceScalarSettings {
    pub colormap: ColorMap,
}

impl Default for FaceScalarSettings {
    fn default() -> Self {
        Self {
            colormap: ColorMap::default(),
        }
    }
}

impl DataUniformBuilder for FaceScalarSettings {
    fn build_uniform(self, device: &wgpu::Device) -> DataUniform {
        self.colormap.get_value().build_uniform(device)
    }

    fn refresh_buffer(self, queue: &mut wgpu::Queue, data_uniform: &DataUniform) {
        self.colormap
            .get_value()
            .refresh_buffer(queue, data_uniform);
    }
}

impl From<&VertexScalarSettings> for VertexScalarSettingsBuffer {
    fn from(settings: &VertexScalarSettings) -> VertexScalarSettingsBuffer {
        VertexScalarSettingsBuffer {
            isoline: settings.isoline,
            colormap: settings.colormap.get_value(),
        }
    }
}

impl Default for VertexScalarSettings {
    fn default() -> Self {
        Self {
            isoline: IsolineSettings::default(),
            colormap: ColorMap::default(),
        }
    }
}

impl DataUniformBuilder for VertexScalarSettings {
    fn build_uniform(self, device: &wgpu::Device) -> DataUniform {
        let settings_buffer: VertexScalarSettingsBuffer = (&self).into();
        settings_buffer.build_uniform(device)
    }

    fn refresh_buffer(self, queue: &mut wgpu::Queue, data_uniform: &DataUniform) {
        let settings_buffer: VertexScalarSettingsBuffer = (&self).into();
        settings_buffer.refresh_buffer(queue, data_uniform);
    }
}

#[non_exhaustive]
pub enum MeshData {
    Color(Vec<[f32; 3]>),
    FaceScalar(Vec<f32>, FaceScalarSettings),
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
            MeshData::FaceScalar(_, uniform) => Some(uniform.build_uniform(device)),
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
            MeshData::FaceScalar(_, uniform) => {
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

pub trait DataUniformBuilder {
    fn build_uniform(self, device: &wgpu::Device) -> DataUniform;
    fn refresh_buffer(self, queue: &mut wgpu::Queue, data_uniform: &DataUniform);
}

impl<T> DataUniformBuilder for T
where
    T: bytemuck::Pod + Copy,
{
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

impl UiDataElement for MeshData {
    fn draw(&mut self, ui: &mut egui::Ui) -> bool {
        use egui::Widget;
        match self {
            MeshData::UVMap(_, data_uniform) | MeshData::UVCornerMap(_, data_uniform) => {
                data_uniform.draw(ui)
            }
            MeshData::VertexScalar(_, data_uniform) => {
                let changed = data_uniform.colormap.draw(ui);
                data_uniform.isoline.draw(ui) || changed
            }
            MeshData::FaceScalar(_, data_uniform) => data_uniform.colormap.draw(ui),
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
            MeshData::FaceScalar(datas, _) => {
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
