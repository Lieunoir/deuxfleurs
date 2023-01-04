use crate::util::BufferDimensions;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(module = "/src/save.js")]
extern "C" {
    fn save_png(filename: &str, data: &[u8]);
}

pub struct Screenshoter {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    output_buffer: wgpu::Buffer,
    buffer_dimensions: BufferDimensions,
    counter: u32,
}

impl Screenshoter {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some("screen texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        let texture = device.create_texture(&desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // we need to store this for later
        let buffer_dimensions = BufferDimensions::new::<u32>(width as usize, height as usize);

        let output_buffer_size = (buffer_dimensions.padded_bytes_per_row * buffer_dimensions.height)
            as wgpu::BufferAddress;
        let output_buffer_desc = wgpu::BufferDescriptor {
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST
                // this tells wpgu that we want to read this buffer from the cpu
                | wgpu::BufferUsages::MAP_READ,
            label: None,
            mapped_at_creation: false,
        };
        let output_buffer = device.create_buffer(&output_buffer_desc);

        Self {
            texture,
            view,
            output_buffer,
            buffer_dimensions,
            counter: 0,
        }
    }

    pub fn get_view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub fn copy_texture_to_buffer(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: std::num::NonZeroU32::new(
                        self.buffer_dimensions.padded_bytes_per_row as u32,
                    ),
                    //rows_per_image: std::num::NonZeroU32::new(self.size.height),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: self.buffer_dimensions.width as u32,
                height: self.buffer_dimensions.height as u32,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn create_png(&mut self, device: &wgpu::Device, submission_index: wgpu::SubmissionIndex) {
        let png_output_path = format!("screenshot_{:03}.png", self.counter);
        // Note that we're not calling `.await` here.
        let buffer_slice = self.output_buffer.slice(..);
        // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
        //let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        let (sender, receiver) = oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        //
        // We pass our submission index so we don't need to wait for any other possible submissions.
        device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index));
        // If a file system is available, write the buffer as a PNG
        if let Ok(Ok(())) = receiver.recv() {
            let data = buffer_slice.get_mapped_range();
            let mut unpadded_data = Vec::<u8>::with_capacity(
                4 * self.buffer_dimensions.width * self.buffer_dimensions.height,
            );
            for chunk in data.chunks(self.buffer_dimensions.padded_bytes_per_row) {
                for cp_data in &chunk[..self.buffer_dimensions.unpadded_bytes_per_row] {
                    unpadded_data.push(*cp_data);
                }
            }

            use image::{ImageBuffer, Rgba};
            let buffer = ImageBuffer::<Rgba<u8>, _>::from_raw(
                self.buffer_dimensions.width as u32,
                self.buffer_dimensions.height as u32,
                unpadded_data,
            )
            .unwrap();
            #[cfg(not(target_arch = "wasm32"))]
            {
                buffer.save(png_output_path).unwrap();
            }
            #[cfg(target_arch = "wasm32")]
            {
                let mut blob = Vec::new();
                buffer.write_to(
                    &mut std::io::Cursor::new(&mut blob),
                    image::ImageOutputFormat::Png,
                );
                save_png(&png_output_path, &blob);
            }
            self.counter += 1;
        }
        self.output_buffer.unmap();
    }
}
