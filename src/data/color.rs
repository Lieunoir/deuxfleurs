use crate::ui::UiDataElement;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColorSettings {
    pub color: [f32; 4],
}

const RED: ColorSettings = ColorSettings {
    color: [0.55, 0.1, 0.1, 1.],
};
const YELLOW: ColorSettings = ColorSettings {
    color: [0.63, 0.63, 0.09, 1.],
};
const WHITE: ColorSettings = ColorSettings {
    color: [0.9, 0.9, 0.9, 1.],
};
const GREEN: ColorSettings = ColorSettings {
    color: [0.1, 0.50, 0.20, 1.],
};
const PINK: ColorSettings = ColorSettings {
    color: [0.50, 0.1, 0.52, 1.],
};
const BLUE: ColorSettings = ColorSettings {
    color: [0.22, 0.22, 0.75, 1.],
};

impl ColorSettings {
    pub fn new(name: &str) -> Self {
        let mut value = 0_u8;
        for &x in name.as_bytes() {
            value = value.wrapping_add(x);
        }
        value = value % 6;
        match value {
            0 => BLUE,
            1 => YELLOW,
            2 => WHITE,
            3 => GREEN,
            4 => PINK,
            5 => RED,
            _ => BLUE,
        }
    }
}

impl Default for ColorSettings {
    fn default() -> Self {
        Self {
            color: [0.2, 0.2, 0.8, 1.],
        }
    }
}

impl UiDataElement for ColorSettings {
    fn draw(&mut self, ui: &mut egui::Ui, _property_changed: &mut bool) -> bool {
        let mut mesh_color = egui::Rgba::from_rgba_unmultiplied(
            self.color[0],
            self.color[1],
            self.color[2],
            self.color[3],
        );
        let mut changed = false;
        ui.horizontal(|ui| {
            changed = egui::widgets::color_picker::color_edit_button_rgba(
                ui,
                &mut mesh_color,
                egui::widgets::color_picker::Alpha::Opaque,
            )
            .changed();
            ui.label("Color");
        });
        self.color = mesh_color.to_array();
        changed
    }
}
