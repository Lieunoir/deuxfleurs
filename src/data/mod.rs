use wgpu::util::DeviceExt;

mod color;
mod colormap;
mod isoline;
mod radius;
mod transform;
mod uv;

pub use color::ColorSettings;
pub(crate) use colormap::ColorMapValues;
pub use colormap::{ColorMap, Colors};
pub use isoline::IsolineSettings;
pub use radius::Radius;
pub use transform::TransformSettings;
pub use uv::UVMapSettings;

pub(crate) struct DataUniform {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub buffer: wgpu::Buffer,
}

pub(crate) trait DataUniformBuilder {
    fn build_uniform(&self, device: &wgpu::Device) -> Option<DataUniform>;
    fn refresh_buffer(&self, queue: &mut wgpu::Queue, data_uniform: &DataUniform);
}

pub(crate) trait DataSettings {
    fn apply_settings(&mut self, other: Self);
}

impl<T> DataUniformBuilder for T
where
    T: bytemuck::Pod + Copy,
{
    fn build_uniform(&self, device: &wgpu::Device) -> Option<DataUniform> {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Data buffer"),
            contents: bytemuck::cast_slice(&[*self]),
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
        Some(DataUniform {
            bind_group_layout,
            bind_group,
            buffer,
        })
    }

    fn refresh_buffer(&self, queue: &mut wgpu::Queue, data_uniform: &DataUniform) {
        queue.write_buffer(&data_uniform.buffer, 0, bytemuck::cast_slice(&[*self]));
    }
}
