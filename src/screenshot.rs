use crate::util::BufferDimensions;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(module = "/src/save.js")]
extern "C" {
    fn save_png(filename: &str, data: &[u8]);
}

pub struct Screenshoter {
    counter: u32,
}

impl Screenshoter {
    pub fn new() -> Self {
        Self { counter: 0 }
    }

    pub fn create_image_buffer(
        &mut self,
        device: &wgpu::Device,
        submission_index: wgpu::SubmissionIndex,
        output_buffer: &wgpu::Buffer,
        buffer_dimensions: &BufferDimensions,
    ) -> Result<Vec<u8>, ()> {
        let buffer_slice = output_buffer.slice(..);
        // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
        //let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        let (sender, receiver) = oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        //
        // We pass our submission index so we don't need to wait for any other possible submissions.
        device.poll(wgpu::PollType::WaitForSubmissionIndex(submission_index));
        // If a file system is available, write the buffer as a PNG
        let res = if let Ok(Ok(())) = receiver.recv() {
            let data = buffer_slice.get_mapped_range();
            let mut unpadded_data =
                Vec::<u8>::with_capacity(4 * buffer_dimensions.width * buffer_dimensions.height);
            for chunk in data.chunks(buffer_dimensions.padded_bytes_per_row) {
                //unpadded_data.extend(&chunk[..self.buffer_dimensions.unpadded_bytes_per_row]);
                for cp_data in chunk[..buffer_dimensions.unpadded_bytes_per_row].chunks_exact(4) {
                    // This is incorrect but necessary, see :
                    // https://erikmcclure.com/blog/everyone-does-srgb-wrong-because/
                    let alpha = 1. - (cp_data[3] as f32) / 255.;
                    let alpha = 1.055 * (alpha.powf(1. / 2.4)) - 0.055;
                    let alpha = ((1. - alpha) * 255.) as u8;
                    unpadded_data.push(cp_data[0]);
                    unpadded_data.push(cp_data[1]);
                    unpadded_data.push(cp_data[2]);
                    unpadded_data.push(alpha);
                }
            }
            Ok(unpadded_data)
        } else {
            Err(())
        };
        output_buffer.unmap();
        res
    }

    pub fn create_png(
        &mut self,
        device: &wgpu::Device,
        submission_index: wgpu::SubmissionIndex,
        output_buffer: &wgpu::Buffer,
        buffer_dimensions: &BufferDimensions,
    ) {
        let png_output_path = format!("screenshot_{:03}.png", self.counter);
        if let Ok(unpadded_data) =
            self.create_image_buffer(device, submission_index, output_buffer, buffer_dimensions)
        {
            use image::{ImageBuffer, Rgba};
            let buffer = ImageBuffer::<Rgba<u8>, _>::from_raw(
                buffer_dimensions.width as u32,
                buffer_dimensions.height as u32,
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
    }
}
