mod color;
mod colormap;
mod isoline;
mod radius;
mod transform;
mod uv;

pub(crate) use color::ColorSettings;
pub(crate) use colormap::ColorMap;
pub(crate) use colormap::ColorMapValues;
pub use colormap::{ColorMapMut, Colors};
pub(crate) use isoline::IsolineSettings;
pub(crate) use radius::Radius;
pub(crate) use transform::TransformSettings;
pub(crate) use uv::UVMapSettings;
pub use uv::UVMapSettingsMut;

pub(crate) mod internal {
    use wgpu::util::DeviceExt;

    pub struct DataUniform {
        pub bind_group_layout: wgpu::BindGroupLayout,
        pub bind_group: wgpu::BindGroup,
        pub buffer: wgpu::Buffer,
    }

    pub trait DataUniformBuilder {
        fn build_uniform(&self, device: &wgpu::Device) -> Option<DataUniform>;
        fn refresh_buffer(&self, queue: &wgpu::Queue, data_uniform: &DataUniform);
    }

    pub trait DataSettings: DataUniformBuilder + Clone {
        fn apply_previous_settings(&mut self, previous: Self);

        fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool;
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
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        fn refresh_buffer(&self, queue: &wgpu::Queue, data_uniform: &DataUniform) {
            queue.write_buffer(&data_uniform.buffer, 0, bytemuck::cast_slice(&[*self]));
        }
    }
}
