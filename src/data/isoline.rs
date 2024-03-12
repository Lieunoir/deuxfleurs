use crate::ui::UiDataElement;
use egui::Widget;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IsolineSettings {
    pub isoline_number: f32,
    _padding: [f32; 3],
}

impl Default for IsolineSettings {
    fn default() -> Self {
        Self {
            isoline_number: 0.,
            _padding: [0.; 3],
        }
    }
}

impl UiDataElement for IsolineSettings {
    fn draw(&mut self, ui: &mut egui::Ui, _property_changed: &mut bool) -> bool {
        egui::Slider::new(&mut self.isoline_number, 0.0..=100.0)
            .text("Isolines")
            .clamp_to_range(false)
            .ui(ui)
            .changed()
    }
}
