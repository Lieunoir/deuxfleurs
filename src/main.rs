use deuxfleurs::resources;
use deuxfleurs::types::SurfaceIndices;
use deuxfleurs::{create_window, StateWrapper};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

fn main() {
    pollster::block_on(run());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    let (event_loop, window) = create_window(1080, 720).await;
    // State::new uses async code, so we're going to wait for it to finish
    let mut state = StateWrapper::new(&event_loop, &window).await;
    /*
    let (banana_v, banana_f) = resources::load_mesh("banana.obj").await.unwrap();
    let mut banana_data = vec![0.; banana_v.len()];
    for (data, vertex) in banana_data.iter_mut().zip(&banana_v) {
        *data = vertex[1];
    }
    state
        .register_mesh("banana", &banana_v, &banana_f)
        .mesh
        .add_vertex_data("test data".into(), banana_data)
        .set_data("test data".into());
        */
    let (spot_v, spot_f) = resources::load_mesh("spot.obj").await.unwrap();
    let spot_f = match spot_f {
        SurfaceIndices::Triangles(t) => t,
        _ => panic!(),
    };
    let mut spot_data_1 = vec![0.; spot_v.len()];
    let mut spot_data_2 = vec![0.; spot_v.len()];
    for (data, vertex) in spot_data_1.iter_mut().zip(&spot_v) {
        *data = vertex[0];
    }
    for (data, vertex) in spot_data_2.iter_mut().zip(&spot_v) {
        *data = vertex[1];
    }
    let mut spot_uv_map = vec![[0., 0.]; spot_v.len()];
    for (data, vertex) in spot_uv_map.iter_mut().zip(&spot_v) {
        *data = [vertex[0] * 0.1, vertex[1] * 0.1];
    }
    let mut spot_uv_mesh = vec![[0., 0., 0.]; spot_v.len()];
    for (data, vertex) in spot_uv_mesh.iter_mut().zip(&spot_v) {
        *data = [vertex[0], vertex[1], 0.];
    }

    let mut spot_corner_uv_map = vec![[0., 0.]; 3 * spot_f.len()];
    for (datas, face) in spot_corner_uv_map.chunks_exact_mut(3).zip(&spot_f) {
        datas[0] = [spot_v[face[0] as usize][0], spot_v[face[0] as usize][1]];
        datas[1] = [spot_v[face[1] as usize][0], spot_v[face[1] as usize][1]];
        datas[2] = [spot_v[face[2] as usize][0], spot_v[face[2] as usize][1]];
    }

    let mut spot_face_scalar = vec![0.; spot_f.len()];
    for (i, face) in spot_face_scalar.iter_mut().enumerate() {
        let value = spot_v[spot_f[i][0] as usize][0];
        *face = value;
    }
    state
        .register_mesh("spot_uv", spot_uv_mesh, spot_f.clone())
        .mesh
        .show_edges(true);
    state
        .register_mesh("spot", spot_v.clone(), spot_f.clone())
        .mesh
        .show_edges(true)
        .add_vertex_scalar("x coord".into(), spot_data_1.clone())
        .add_vertex_scalar("y coord".into(), spot_data_2)
        .add_uv_map("uv".into(), spot_uv_map)
        .add_corner_uv_map("corner uv".into(), spot_corner_uv_map)
        .add_face_scalar("face scalar".into(), spot_face_scalar)
        .set_data(Some("y coord".into()));

    let mut curves = Vec::new();
    for f in &spot_f {
        if !curves.contains(&[f[0], f[1]]) && !curves.contains(&[f[1], f[0]]) {
            curves.push([f[0], f[1]]);
        }
        if !curves.contains(&[f[2], f[1]]) && !curves.contains(&[f[1], f[2]]) {
            curves.push([f[2], f[1]]);
        }
        if !curves.contains(&[f[0], f[2]]) && !curves.contains(&[f[2], f[0]]) {
            curves.push([f[0], f[2]]);
        }
    }
    //curves.push([(spot_v.len() - i) as u32, 100]);
    //curves.push([i as u32, 100]);
    state
        .register_curve("spot_c".into(), spot_v.clone(), curves)
        //.register_point_cloud("spot_pc".into(), spot_v.clone())
        .add_scalar("x coord".into(), spot_data_1)
        .set_data(Some("x coord".into()))
        ;

    state
        .register_point_cloud("spot_pc".into(), spot_v.clone())
        .set_radius(0.2)
        //.add_scalar("x coord".into(), spot_data_1)
        //.set_data(Some("x coord".into()))
        ;

    let mut last_selected = 0;
    let mut last_selected_mesh = "".into();
    state.set_callback(move |ui, state| {
        ui.label("User defined stuff here : ");
        if ui
            .add(egui::Button::new("Filter input mesh (require mesh loaded)"))
            .clicked()
        {
            let mut vertices = Vec::new();
            let mut indices = Vec::new();
            let mut filtered = false;
            if let Some(model) = state.get_model("loaded mesh") {
                //let mut vertices = model.mesh.vertices.clone();
                vertices = vec![[0., 0., 0.]; model.mesh.vertices.len()];
                for (vertex, orig_vertex) in vertices.iter_mut().zip(model.mesh.vertices.iter()) {
                    *vertex = *orig_vertex;
                }
                indices = match &model.mesh.indices {
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
                state.register_mesh("loaded mesh", new_pos, indices);
            }
        }

        if let Some((model_name, item)) = state.get_picked().clone() {
            if last_selected != item || last_selected_mesh != *model_name {
                if let Some(model) = state.get_model(&model_name) {
                    let n_v = model.mesh.vertices.len();
                    if item < n_v {
                        let mut selected = vec![0.; n_v];
                        last_selected = item;
                        last_selected_mesh = model_name.clone();
                        selected[item] = 1.;
                        model
                            .mesh
                            .add_vertex_scalar("selected vertex".into(), selected);
                    } else {
                        let mut selected = vec![0.; model.mesh.indices.size()];
                        last_selected = item;
                        last_selected_mesh = model_name.clone();
                        selected[item - n_v] = 1.;
                        model.mesh.add_face_scalar("selected face".into(), selected);
                    }
                }
            }
        }
    });
    state.run(event_loop, &window);
}
