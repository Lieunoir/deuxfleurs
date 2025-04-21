use deuxfleurs::types::SurfaceIndices;
use deuxfleurs::{load_mesh, Settings, State, StateBuilder};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

fn main() {
    pollster::block_on(run());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    let (spot_v, spot_f) = load_mesh("examples/assets/spot.obj").await.unwrap();
    let init = move |state: &mut State| {
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
        let surface1 = state.register_surface("spot_uv".into(), spot_uv_mesh, spot_f.clone());
        surface1.show_edges(true);
        surface1.add_vertex_scalar("x coord".into(), spot_data_1.clone());
        let surface2 = state
            .register_surface("spot".into(), spot_v.clone(), spot_f.clone())
            .show_edges(true);
        surface2.add_vertex_scalar("x coord".into(), spot_data_1.clone());
        surface2
            .add_vertex_vector_field("positions".into(), spot_v.clone())
            .set_magnitude(0.1);
        surface2.add_vertex_scalar("y coord".into(), spot_data_2);
        surface2.add_uv_map("uv".into(), spot_uv_map);
        surface2.add_corner_uv_map("corner uv".into(), spot_corner_uv_map);
        surface2.add_face_scalar("face scalar".into(), spot_face_scalar);
        surface2.set_data(Some("y coord".into()));

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
        let curve = state.register_segment("spot_c".into(), spot_v.clone(), curves);
        //.register_point_cloud("spot_pc".into(), spot_v.clone())
        curve.add_scalar("x coord".into(), spot_data_1.clone());
        curve.set_data(Some("x coord".into()));

        let pc = state.register_point_cloud("spot_pc".into(), spot_v.clone());
        pc.add_scalar("x coord".into(), spot_data_1);
        pc.set_radius(0.02)
        //.set_data(Some("x coord".into()))
        ;
    };

    let callback = move |_ui: &mut egui::Ui, _state: &mut State| {};
    StateBuilder::run(
        1080,
        720,
        Some("deuxfleurs".into()),
        Settings::default(),
        init,
        callback,
    );
}
