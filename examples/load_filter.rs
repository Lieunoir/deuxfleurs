use deuxfleurs::types::SurfaceIndices;
use deuxfleurs::ui::LoadObjButton;
use deuxfleurs::{Color, State, StateBuilder};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

fn main() {
    pollster::block_on(run());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    let init = move |_state: &mut State| {};

    let callback = move |ui: &mut egui::Ui, state: &mut State| {
        ui.label("User defined stuff here : ");
        ui.add(LoadObjButton::new("Load obj", "loaded mesh", state));
        if ui
            .add(egui::Button::new("Filter input mesh (require mesh loaded)"))
            .clicked()
        {
            let mut vertices = Vec::new();
            let mut indices = Vec::new();
            let mut filtered = false;
            if let Some(surface) = state.get_surface("loaded mesh") {
                //let mut vertices = surface.geometry.vertices.clone();
                vertices = vec![[0., 0., 0.]; surface.geometry().vertices.len()];
                for (vertex, orig_vertex) in
                    vertices.iter_mut().zip(surface.geometry().vertices.iter())
                {
                    *vertex = *orig_vertex;
                }
                indices = match &surface.geometry().indices {
                    SurfaceIndices::Triangles(t) => t.clone(),
                    _ => panic!(),
                };
                filtered = true;
            }
            if filtered {
                let mut weights = vec![0; vertices.len()];
                //let mut new_pos = vertices.clone();
                let mut new_pos = vec![[0., 0., 0.]; vertices.len()];
                for face in &indices {
                    let v1 = vertices[face[0] as usize];
                    let v2 = vertices[face[1] as usize];
                    let v3 = vertices[face[2] as usize];
                    for index in face {
                        weights[*index as usize] += 3;
                        new_pos[*index as usize] = [
                            new_pos[*index as usize][0] + v1[0] + v2[0] + v3[0],
                            new_pos[*index as usize][1] + v1[1] + v2[1] + v3[1],
                            new_pos[*index as usize][2] + v1[2] + v2[2] + v3[2],
                        ];
                    }
                }
                for (vertex, weight) in new_pos.iter_mut().zip(weights) {
                    for i in 0..3 {
                        vertex[i] = vertex[i] / (weight as f32);
                    }
                }
                state.register_surface("loaded mesh".into(), new_pos, indices);
            }
        }
    };
    StateBuilder::run(
        1080,
        720,
        Some("deuxfleurs".into()),
        deuxfleurs::Settings {
            color: Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
            ..Default::default()
        },
        init,
        callback,
    );
}
