use crate::ui::UiDataElement;
use egui::Widget;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Radius {
    pub radius: f32,
    _padding: [u32; 3],
}

impl Default for Radius {
    fn default() -> Self {
        Self {
            radius: 0.01,
            _padding: [0; 3],
        }
    }
}

impl UiDataElement for Radius {
    fn draw(&mut self, ui: &mut egui::Ui, property_changed: &mut bool) -> bool {
        egui::Slider::new(&mut self.radius, 0.1..=100.0)
            .text("Radius")
            .clamp_to_range(false)
            .logarithmic(true)
            .ui(ui)
            .changed()
    }
}
