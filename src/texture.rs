use crate::util::BufferDimensions;

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
pub const ALBEDO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub const NORMALS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
// Same as albedo
pub const PICKER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub const SSAO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
pub const SCREENSHOT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

pub struct TextureBufferPool {
    size: wgpu::Extent3d,
    // concerned by super sampling
    // custom formats
    // picking should be concerned by aa, so could be moved?
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    albedo_or_picking: wgpu::Texture,
    picking_view: wgpu::TextureView,
    albedo_view: wgpu::TextureView,
    normals: wgpu::Texture,
    normals_view: wgpu::TextureView,

    // Could be used as blend_render_target too using uniform alpha?
    // not concerned by super sampling
    // screenshot has to be rgba format
    screenshot_or_blend_stored: wgpu::Texture,
    blend_stored_view: wgpu::TextureView,
    screenshot_view: wgpu::TextureView,
    // possibly concerned by super sampling
    // has to be same format as window
    blend_render_target: wgpu::Texture,
    blend_render_target_view: wgpu::TextureView,
    output_buffer: wgpu::Buffer,
    output_buffer_dimensions: BufferDimensions,
    // not concerned by aa
    // own format
    ssao_view: wgpu::TextureView,
    denoiser_edges_view: wgpu::TextureView,
    denoised_ssao: wgpu::Texture,
    denoised_ssao_view: wgpu::TextureView,
    history_ssao: wgpu::Texture,
    history_ssao_view: wgpu::TextureView,
}

impl TextureBufferPool {
    pub fn new(
        device: &wgpu::Device,
        texture_size: wgpu::Extent3d,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let half_size = wgpu::Extent3d {
            width: (texture_size.width / 2).max(1),
            height: (texture_size.height / 2).max(1),
            depth_or_array_layers: 1,
        };
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let albedo_or_picking = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC,
            label: Some("picking_or_albedo_texture"),
            view_formats: &[],
        });
        let albedo_view = albedo_or_picking.create_view(&wgpu::TextureViewDescriptor::default());
        let picking_view = albedo_or_picking.create_view(&wgpu::TextureViewDescriptor::default());
        let normals = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("normals_pbr_texture"),
            view_formats: &[],
        });
        let normals_view = normals.create_view(&wgpu::TextureViewDescriptor::default());
        let screenshot_or_blend_stored = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SCREENSHOT_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC,
            label: Some("screenshot_or_blend_texture"),
            view_formats: &[],
        });
        let screenshot_view =
            screenshot_or_blend_stored.create_view(&wgpu::TextureViewDescriptor::default());
        let blend_stored_view =
            screenshot_or_blend_stored.create_view(&wgpu::TextureViewDescriptor::default());
        let blend_render_target = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("blend_render_target_texture"),
            view_formats: &[],
        });
        let blend_render_target_view =
            blend_render_target.create_view(&wgpu::TextureViewDescriptor::default());

        let ssao = device.create_texture(&wgpu::TextureDescriptor {
            //size: half_size,
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SSAO_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("ssao_texture"),
            view_formats: &[],
        });
        let ssao_view = ssao.create_view(&wgpu::TextureViewDescriptor::default());
        let denoiser_edges = device.create_texture(&wgpu::TextureDescriptor {
            //size: half_size,
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SSAO_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("denoised ssao_texture"),
            view_formats: &[],
        });
        let denoiser_edges_view =
            denoiser_edges.create_view(&wgpu::TextureViewDescriptor::default());
        let denoised_ssao = device.create_texture(&wgpu::TextureDescriptor {
            //size: half_size,
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SSAO_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC,
            label: Some("denoised ssao_texture"),
            view_formats: &[],
        });
        let denoised_ssao_view = denoised_ssao.create_view(&wgpu::TextureViewDescriptor::default());

        let history_ssao = device.create_texture(&wgpu::TextureDescriptor {
            //size: half_size,
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SSAO_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST,
            label: Some("history_ssao_texture"),
            view_formats: &[],
        });
        let history_ssao_view = history_ssao.create_view(&wgpu::TextureViewDescriptor::default());

        let output_buffer_dimensions =
            BufferDimensions::new::<u32>(texture_size.width as usize, texture_size.height as usize);

        let output_buffer_size = (output_buffer_dimensions.padded_bytes_per_row
            * output_buffer_dimensions.height)
            as wgpu::BufferAddress;
        let output_buffer_desc = wgpu::BufferDescriptor {
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            label: Some("output_buffer"),
            mapped_at_creation: false,
        };
        let output_buffer = device.create_buffer(&output_buffer_desc);
        Self {
            size: texture_size,
            albedo_or_picking,
            albedo_view,
            picking_view,
            normals,
            normals_view,
            screenshot_or_blend_stored,
            screenshot_view,
            blend_stored_view,
            blend_render_target,
            blend_render_target_view,
            output_buffer,
            output_buffer_dimensions,
            depth,
            depth_view,
            ssao_view,
            denoiser_edges_view,
            denoised_ssao,
            denoised_ssao_view,
            history_ssao,
            history_ssao_view,
        }
    }

    pub fn get_albedo_view(&self) -> &wgpu::TextureView {
        &self.albedo_view
    }

    pub fn get_normals_view(&self) -> &wgpu::TextureView {
        &self.normals_view
    }

    pub fn get_blend_target_view(&self) -> &wgpu::TextureView {
        &self.blend_render_target_view
    }

    pub fn get_blend_stored_view(&self) -> &wgpu::TextureView {
        &self.blend_stored_view
    }

    pub fn get_picker_view(&self) -> &wgpu::TextureView {
        &self.picking_view
    }

    pub fn get_output_buffer(&self) -> &wgpu::Buffer {
        &self.output_buffer
    }

    pub fn get_output_buffer_dimensions(&self) -> &BufferDimensions {
        &self.output_buffer_dimensions
    }

    pub fn get_depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    pub fn get_ssao_view(&self) -> &wgpu::TextureView {
        &self.ssao_view
    }

    pub fn get_denoised_ssao_view(&self) -> &wgpu::TextureView {
        &self.denoised_ssao_view
    }

    pub fn get_denoiser_edges_view(&self) -> &wgpu::TextureView {
        &self.denoiser_edges_view
    }

    pub fn get_history_ssao_view(&self) -> &wgpu::TextureView {
        &self.history_ssao_view
    }

    pub fn get_denoised_ssao(&self) -> &wgpu::Texture {
        &self.denoised_ssao
    }

    pub fn get_history_ssao(&self) -> &wgpu::Texture {
        &self.history_ssao
    }

    pub fn get_ssao_size(&self) -> &wgpu::Extent3d {
        &self.size
    }

    pub fn copy_screenshot_texture_to_buffer(&mut self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                aspect: wgpu::TextureAspect::All,
                texture: &self.screenshot_or_blend_stored,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.output_buffer_dimensions.padded_bytes_per_row as u32),
                    //rows_per_image: std::num::NonZeroU32::new(self.size.height),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: self.output_buffer_dimensions.width as u32,
                height: self.output_buffer_dimensions.height as u32,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn copy_picker_texture_to_buffer(&mut self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                aspect: wgpu::TextureAspect::All,
                texture: &self.albedo_or_picking,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.output_buffer_dimensions.padded_bytes_per_row as u32),
                    //rows_per_image: std::num::NonZeroU32::new(self.size.height),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: self.output_buffer_dimensions.width as u32,
                height: self.output_buffer_dimensions.height as u32,
                depth_or_array_layers: 1,
            },
        );
    }
}
