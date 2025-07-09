use crate::ui::UiDataElement;
use egui::Widget;
#[cfg(feature = "saves")]
use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[cfg_attr(feature = "saves", derive(Serialize, Deserialize))]
pub struct Radius {
    radius: f32,
    characteristic_length: f32,
    #[cfg_attr(feature = "saves", serde(skip))]
    _padding: [u32; 2],
}

impl Radius {
    pub(crate) fn new(characteristic_length: f32) -> Self {
        Self {
            radius: 1.,
            characteristic_length,
            _padding: [0; 2],
        }
    }

    pub(crate) fn get_relative(&self) -> f32 {
        self.radius
    }

    pub(crate) fn get_absolute(&self) -> f32 {
        self.characteristic_length * self.radius
    }

    pub(crate) fn set_relative(&mut self, l: f32) {
        self.radius = l;
    }

    pub(crate) fn set_absolute(&mut self, l: f32) {
        self.radius = l / self.characteristic_length;
    }
}

impl UiDataElement for Radius {
    fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            changed = egui::DragValue::new(&mut self.radius)
                .prefix("Radius: ")
                .speed(0.1)
                .ui(ui)
                .changed();
        });
        changed
    }
}
