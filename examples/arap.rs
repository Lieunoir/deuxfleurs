use deuxfleurs::{
    load_mesh_blocking,
    picker::{Picked, SurfacePicked},
    types::SurfaceIndices,
};
use faer::{
    MatMut, Row,
    prelude::{ReborrowMut, Solve},
    sparse::{
        SparseColMat, SymbolicSparseColMat,
        linalg::solvers::{Llt, SymbolicLlt},
    },
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::prelude::ParallelSlice;
use std::{
    collections::VecDeque,
    sync::{
        Mutex,
        atomic::{AtomicI8, AtomicU32, Ordering},
    },
};

fn main() {
    faer::set_global_parallelism(faer::Par::Seq);
    let (spot_v, spot_f) = load_mesh_blocking("examples/assets/spot.obj".into()).unwrap();
    let spot_f = match spot_f {
        SurfaceIndices::Triangles(t) => t,
        _ => panic!(),
    };
    let mut arap = Arap::new(&spot_v, &spot_f);
    let mut handle = deuxfleurs::init();
    handle.register_surface("Spot", spot_v, spot_f);

    let handle = handle.with_callback(|ui, state| {
        ui.heading("Arap edition");

        ui.label("First select a vertex, then click on the add constraint button.");

        if let Some((_, Picked::Surface(SurfacePicked::Vertex(item)))) = state.get_picked().clone()
        {
            if ui.button("Add Constraint").clicked() {
                let mut surface = state.get_surface_mut("Spot").unwrap();
                arap.set_constraint(item);
                surface.add_vertex_points("Constraints", arap.get_contrainted_vertices().to_vec());
            }
        } else {
            ui.add_enabled(false, egui::Button::new("Add Constraint"));
        }

        ui.label(
            "Once at least two constraints are set: select a constrained vertex, \
            check 'Edition Gizmo' and move the vertex using the gizmo.",
        );

        // Iterate the solver
        if arap.get_contrainted_vertices().len() >= 2 {
            let surface = state.get_surface("Spot").unwrap();
            let vertices = arap.solve(&surface.geometry().vertices, true).unwrap();
            let indices = surface.geometry().indices.clone();
            state.register_surface("Spot", vertices, indices);
        }
    });

    handle.run(1080, 720, Some("deuxfleurs"));
}

struct Arap {
    constraints_indices: Vec<u32>,
    laplacian: SparseColMat<u32, f32>,
    rotations: Vec<[f32; 9]>,
    original_vertices: Vec<[f32; 3]>,
    solver: SymbolicLlt<u32>,
    solving: bool,
}

fn apply_matrix(matrix: &[f32; 9], vector: &[f32; 3]) -> [f32; 3] {
    let mut res = [0.; 3];
    for (row, res) in matrix.chunks_exact(3).zip(&mut res) {
        for (m, v) in row.iter().zip(vector) {
            *res += m * v;
        }
    }
    res
}

fn add_matrices(m_1: &[f32; 9], m_2: &[f32; 9]) -> [f32; 9] {
    let mut res = [0.; 9];
    for ((m_1, m_2), res) in m_1.iter().zip(m_2).zip(&mut res) {
        *res = m_1 + m_2;
    }
    res
}

impl Arap {
    fn new(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> Self {
        let nv = vertices.len();
        let (l, f, e) = build_intrinsic_delaunay(&vertices, &faces);
        let laplacian = build_cotan_laplacian_intrinsic(&l, &f, &e, vertices.len());

        let constraints_indices = Vec::new();
        let rotations = vec![[1., 0., 0., 0., 1., 0., 0., 0., 1.]; nv];
        let original_vertices = vertices
            .iter()
            .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
            .collect();
        let solver = SymbolicLlt::try_new(laplacian.symbolic(), faer::Side::Lower).unwrap();
        Self {
            constraints_indices,
            laplacian,
            rotations,
            original_vertices,
            solving: true,
            solver,
        }
    }

    fn set_constraint(&mut self, index: u32) {
        if let Some(position) = self.constraints_indices.iter().position(|i| *i == index) {
            self.constraints_indices.remove(position);
        } else {
            self.constraints_indices.push(index);
        }
    }

    fn get_contrainted_vertices(&self) -> &[u32] {
        &self.constraints_indices
    }

    // Current vertices are needed for constraints
    fn solve(&mut self, vertices: &[[f32; 3]], force: bool) -> Option<Vec<[f32; 3]>> {
        if force || self.solving {
            // Updates positions

            //let (sym, v) = self.laplacian.parts();
            let rhs: Vec<_> = (0..self.original_vertices.len())
                .map(|i| {
                    let mut res = [0.; 3];
                    let (indices, vals) = self.laplacian.rb_mut().idx_val_of_col_mut(i);
                    for (j, value) in indices.zip(vals) {
                        if i != j {
                            let diff_eges = [
                                *value
                                    * 0.5
                                    * (self.original_vertices[j][0] - self.original_vertices[i][0]),
                                *value
                                    * 0.5
                                    * (self.original_vertices[j][1] - self.original_vertices[i][1]),
                                *value
                                    * 0.5
                                    * (self.original_vertices[j][2] - self.original_vertices[i][2]),
                            ];
                            let matrix = add_matrices(&self.rotations[i], &self.rotations[j]);
                            let res_temp = apply_matrix(&matrix, &diff_eges);
                            res[0] += res_temp[0];
                            res[1] += res_temp[1];
                            res[2] += res_temp[2];
                        }
                    }
                    res
                })
                .collect();
            let mut rhs =
                faer::Mat::<f32>::from_fn(self.original_vertices.len(), 3, |i, j| rhs[i][j]);

            let mut lap = self.laplacian.clone();
            let c_values = self
                .constraints_indices
                .iter()
                .map(|i| Row::from_fn(3, |j| vertices[*i as usize][j] as f32))
                .collect::<Vec<_>>();
            apply_constraints_mat(&mut lap, rhs.as_mut(), &self.constraints_indices, &c_values);
            let llt =
                Llt::try_new_with_symbolic(self.solver.clone(), lap.as_ref(), faer::Side::Lower)
                    .unwrap();
            llt.solve_in_place(rhs.as_mut());

            // Update rotations from positions
            (0..self.original_vertices.len())
                .into_par_iter()
                .map(|i| {
                    let mut a0 = 0.;
                    let mut a1 = 0.;
                    let mut a2 = 0.;
                    let mut a3 = 0.;
                    let mut a4 = 0.;
                    let mut a5 = 0.;
                    let mut a6 = 0.;
                    let mut a7 = 0.;
                    let mut a8 = 0.;
                    let (sym, values) = self.laplacian.parts();
                    let indices = sym.row_idx_of_col(i);
                    let range = sym.col_range(i);
                    let vals = &values[range];
                    //let (indices, vals) = self.laplacian.rb_mut().idx_val_of_col_mut(i);
                    for (j, value) in indices.zip(vals) {
                        if i != j {
                            let orig_edge = [
                                self.original_vertices[j][0] - self.original_vertices[i][0],
                                self.original_vertices[j][1] - self.original_vertices[i][1],
                                self.original_vertices[j][2] - self.original_vertices[i][2],
                            ];
                            let new_edge = rhs.row(j) - rhs.row(i);
                            let new_edge = [new_edge[0], new_edge[1], new_edge[2]];
                            a0 += (*value * (orig_edge[0] * new_edge[0])) as f32;
                            a1 += (*value * (orig_edge[0] * new_edge[1])) as f32;
                            a2 += (*value * (orig_edge[0] * new_edge[2])) as f32;
                            a3 += (*value * (orig_edge[1] * new_edge[0])) as f32;
                            a4 += (*value * (orig_edge[1] * new_edge[1])) as f32;
                            a5 += (*value * (orig_edge[1] * new_edge[2])) as f32;
                            a6 += (*value * (orig_edge[2] * new_edge[0])) as f32;
                            a7 += (*value * (orig_edge[2] * new_edge[1])) as f32;
                            a8 += (*value * (orig_edge[2] * new_edge[2])) as f32;
                        }
                    }
                    let mut u0 = 0.;
                    let mut u1 = 0.;
                    let mut u2 = 0.;
                    let mut u3 = 0.;
                    let mut u4 = 0.;
                    let mut u5 = 0.;
                    let mut u6 = 0.;
                    let mut u7 = 0.;
                    let mut u8 = 0.;
                    let mut v0 = 0.;
                    let mut v1 = 0.;
                    let mut v2 = 0.;
                    let mut v3 = 0.;
                    let mut v4 = 0.;
                    let mut v5 = 0.;
                    let mut v6 = 0.;
                    let mut v7 = 0.;
                    let mut v8 = 0.;
                    fast_svd_3x3::svd_mat(
                        &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut a6, &mut a7,
                        &mut a8, &mut u0, &mut u1, &mut u2, &mut u3, &mut u4, &mut u5, &mut u6,
                        &mut u7, &mut u8, &mut v0, &mut v1, &mut v2, &mut v3, &mut v4, &mut v5,
                        &mut v6, &mut v7, &mut v8,
                    );
                    // Result is u transposed times v
                    // Transposed u
                    let u2 = if a8 < 0. { -u2 } else { u2 };
                    let u5 = if a8 < 0. { -u5 } else { u5 };
                    let u8 = if a8 < 0. { -u8 } else { u8 };
                    let res_u = [-u0, -u3, -u6, -u1, -u4, -u7, -u2, -u5, -u8];
                    let res_v = [v0, v1, v2, v3, v4, v5, v6, v7, v8];
                    let mut res = [0.0_f32; 9];
                    for i in 0..3 {
                        for j in 0..3 {
                            let v = res_v[i * 3] * res_u[j]
                                + res_v[i * 3 + 1] * res_u[j + 3]
                                + res_v[i * 3 + 2] * res_u[j + 6];
                            res[i * 3 + j] = v as f32;
                        }
                    }
                    res
                })
                .collect_into_vec(&mut self.rotations);

            Some(
                rhs.row_iter()
                    .map(|row| [row[0] as f32, row[1] as f32, row[2] as f32])
                    .collect(),
            )
        } else {
            None
        }
    }
}

trait MyIndex: Sized {
    type Atomic: Sync;

    fn slice_to_atomic_slice(v: &mut [Self]) -> &[Self::Atomic];

    fn write_to_atomic(self, atomic: &Self::Atomic);
}

impl MyIndex for u32 {
    type Atomic = AtomicU32;
    fn slice_to_atomic_slice(v: &mut [Self]) -> &[Self::Atomic] {
        let [] = [(); align_of::<Self::Atomic>() - align_of::<Self>()];
        unsafe { &*(v as *mut [Self] as *const [Self::Atomic]) }
    }

    fn write_to_atomic(self, atomic: &Self::Atomic) {
        atomic.store(self, std::sync::atomic::Ordering::Relaxed);
    }
}

fn atomic_from_mut_slice_f32(v: &mut [f32]) -> &[AtomicU32] {
    let [] = [(); align_of::<AtomicU32>() - align_of::<f32>()];
    unsafe { &*(v as *mut [f32] as *const [AtomicU32]) }
}

fn apply_constraints_mat(
    l: &mut SparseColMat<u32, f32>,
    mut rhs: MatMut<f32>,
    c_indices: &[u32],
    c_values: &[Row<f32>],
) {
    let (sym, v) = l.parts_mut();
    let indices = sym.row_idx();
    indices
        .par_iter()
        .zip(v.par_iter_mut())
        .for_each(|(i, value)| {
            if c_indices.contains(&i) {
                *value = 0.;
            }
        });
    for (index, c_value) in c_indices.iter().zip(c_values) {
        let (sym, v) = l.parts_mut();
        let indices = sym.row_idx_of_col_raw(*index as usize);
        let range = sym.col_range(*index as usize);
        for (value, i) in v[range].iter_mut().zip(indices) {
            if *i != *index {
                rhs.rb_mut()
                    .row_mut(*i as usize)
                    .iter_mut()
                    .zip(c_value.iter())
                    .for_each(|(v, c)| *v -= *value * c);
                *value = 0.;
            } else {
                rhs.rb_mut()
                    .row_mut(*i as usize)
                    .iter_mut()
                    .zip(c_value.iter())
                    .for_each(|(v, c)| *v = *c);
                *value = 1.;
            }
        }
    }
}

fn is_delaunay(l_s: f32, l11: f32, l12: f32, l21: f32, l22: f32) -> bool {
    let tan_1 = f32::sqrt(
        ((l_s - l11 + l12) * (l_s - l12 + l11)) / ((l_s + l11 + l12) * (-l_s + l11 + l12)),
    );
    let tan_2 = f32::sqrt(
        ((l_s - l21 + l22) * (l_s - l22 + l21)) / ((l_s + l21 + l22) * (-l_s + l21 + l22)),
    );
    let cot_1 = (1. - tan_1 * tan_1) / (2. * tan_1);
    let cot_2 = (1. - tan_2 * tan_2) / (2. * tan_2);
    cot_1 + cot_2 >= 0.
}

fn build_intrinsic_delaunay(
    v: &[[f32; 3]],
    f: &[[u32; 3]],
) -> (Vec<[f32; 3]>, Vec<[u32; 3]>, Vec<([u32; 3], [i8; 3])>) {
    #[cfg(not(target_arch = "wasm32"))]
    let mut f = f.to_owned();
    struct Face<I> {
        v: [I; 3],
        adj_f: [I; 3],
        adj_f_i: [i8; 3],
        l: [f32; 3],
    }

    let mut e = build_edge_map(&f, v.len());

    let mut lengths = {
        let mut res = Vec::new();
        f.par_iter()
            .map(|row| {
                let v1 = &v[row[0] as usize];
                let v2 = &v[row[1] as usize];
                let v3 = &v[row[2] as usize];
                let edge1 = [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]];
                let edge2 = [v3[0] - v2[0], v3[1] - v2[1], v3[2] - v2[2]];
                let edge3 = [v1[0] - v3[0], v1[1] - v3[1], v1[2] - v3[2]];
                [
                    f32::sqrt(edge1[0] * edge1[0] + edge1[1] * edge1[1] + edge1[2] * edge1[2]),
                    f32::sqrt(edge2[0] * edge2[0] + edge2[1] * edge2[1] + edge2[2] * edge2[2]),
                    f32::sqrt(edge3[0] * edge3[0] + edge3[1] * edge3[1] + edge3[2] * edge3[2]),
                ]
            })
            .collect_into_vec(&mut res);
        res
    };

    let mut faces = Vec::new();
    f.par_iter()
        .zip(lengths.par_iter())
        .zip(e.par_iter())
        .map(|((f, l), e)| {
            //for ((f, l), e) in f.rows().into_iter().zip(lengths.iter()).zip(e.iter()) {
            Face {
                v: [f[0], f[1], f[2]],
                adj_f: e.0,
                adj_f_i: e.1,
                l: *l,
            }
        })
        .collect_into_vec(&mut faces);

    let mut edges_queue = VecDeque::with_capacity(faces.len());
    let mut edges_marked = vec![false; 3 * faces.len()];

    for (face_index, face) in faces.iter().enumerate() {
        for i in 0..3 {
            if face.adj_f_i[i] >= 0 && face_index < face.adj_f[i] as usize {
                edges_queue.push_back((face_index as u32, i as u8));
                edges_marked[3 * face_index + i] = true;
            }
        }
    }

    while let Some((face_index, edge_index)) = edges_queue.pop_back() {
        edges_marked[3 * face_index as usize + edge_index as usize] = false;
        let f1 = &faces[face_index as usize];
        let opp_f = f1.adj_f[edge_index as usize];
        let opp_f_i = f1.adj_f_i[edge_index as usize];
        if opp_f_i >= 0 && face_index != opp_f {
            let e_i_1 = edge_index;
            let e_i_2 = if edge_index + 1 < 3 {
                edge_index + 1
            } else {
                edge_index - 2
            };
            let e_i_3 = if edge_index + 2 < 3 {
                edge_index + 2
            } else {
                edge_index - 1
            };
            let l_s = f1.l[e_i_1 as usize];
            let l11 = f1.l[e_i_2 as usize];
            let l12 = f1.l[e_i_3 as usize];
            let f2 = &faces[opp_f as usize];
            let oe_i_1 = opp_f_i as usize;
            let oe_i_2 = if opp_f_i + 1 < 3 {
                oe_i_1 + 1
            } else {
                oe_i_1 - 2
            };
            let oe_i_3 = if opp_f_i + 2 < 3 {
                oe_i_1 + 2
            } else {
                oe_i_1 - 1
            };
            let l21 = f2.l[oe_i_2];
            let l22 = f2.l[oe_i_3];
            if !is_delaunay(l_s, l11, l12, l21, l22) {
                let f11 = f1.adj_f[e_i_2 as usize];
                let f12 = f1.adj_f[e_i_3 as usize];
                let f21 = f2.adj_f[oe_i_2 as usize];
                let f22 = f2.adj_f[oe_i_3 as usize];
                let i11 = f1.adj_f_i[e_i_2 as usize];
                let i12 = f1.adj_f_i[e_i_3 as usize];
                let i21 = f2.adj_f_i[oe_i_2 as usize];
                let i22 = f2.adj_f_i[oe_i_3 as usize];
                let v11 = f1.v[e_i_2 as usize];
                let v12 = f1.v[e_i_3 as usize];
                let v21 = f2.v[oe_i_2 as usize];
                let v22 = f2.v[oe_i_3 as usize];
                let tan_a_2 = f32::sqrt(
                    ((l12 - l11 + l_s) * (l12 - l_s + l11))
                        / ((l12 + l11 + l_s) * (l_s + l11 - l12)),
                );
                let tan_d_2 = f32::sqrt(
                    ((l21 - l22 + l_s) * (l21 - l_s + l22))
                        / ((l21 + l22 + l_s) * (l_s + l22 - l21)),
                );
                let tan_s = (tan_a_2 + tan_d_2) / (1. - tan_a_2 * tan_d_2);
                let cos = (1. - tan_s * tan_s) / (1. + tan_s * tan_s);
                let new_l = f32::sqrt(l11 * l11 + l22 * l22 - 2. * l11 * l22 * cos);
                let mut new_f_1 = match e_i_1 {
                    2 => Face {
                        v: [v11, v12, v22],
                        l: [l11, new_l, l22],
                        adj_f: [f11, opp_f, f22],
                        adj_f_i: [i11, oe_i_3 as i8, i22],
                    },
                    1 => Face {
                        v: [v12, v22, v11],
                        l: [new_l, l22, l11],
                        adj_f: [opp_f, f22, f11],
                        adj_f_i: [oe_i_3 as i8, i22, i11],
                    },
                    _ => Face {
                        v: [v22, v11, v12],
                        l: [l22, l11, new_l],
                        adj_f: [f22, f11, opp_f],
                        adj_f_i: [i22, i11, oe_i_3 as i8],
                    },
                };
                let mut new_f_2 = match oe_i_1 {
                    2 => Face {
                        v: [v21, v22, v12],
                        l: [l21, new_l, l12],
                        adj_f: [f21, face_index, f12],
                        adj_f_i: [i21, e_i_3 as i8, i12],
                    },
                    1 => Face {
                        v: [v22, v12, v21],
                        l: [new_l, l12, l21],
                        adj_f: [face_index, f12, f21],
                        adj_f_i: [e_i_3 as i8, i12, i21],
                    },
                    _ => Face {
                        v: [v12, v21, v22],
                        l: [l12, l21, new_l],
                        adj_f: [f12, f21, face_index],
                        adj_f_i: [i12, i21, e_i_3 as i8],
                    },
                };
                if f22 == face_index {
                    new_f_1.adj_f_i[e_i_1 as usize] = e_i_1 as i8;
                } else if f22 == opp_f {
                    new_f_1.adj_f_i[e_i_1 as usize] = e_i_1 as i8;
                    new_f_1.adj_f[e_i_1 as usize] = face_index;
                }
                if f11 == face_index {
                    new_f_1.adj_f_i[e_i_2 as usize] = e_i_2 as i8;
                } else if f11 == opp_f {
                    new_f_1.adj_f_i[e_i_2 as usize] = e_i_2 as i8;
                    new_f_1.adj_f[e_i_2 as usize] = face_index;
                }
                if f12 == opp_f {
                    new_f_2.adj_f_i[oe_i_1 as usize] = oe_i_1 as i8;
                } else if f12 == face_index {
                    new_f_2.adj_f_i[oe_i_1 as usize] = oe_i_1 as i8;
                    new_f_2.adj_f[oe_i_1 as usize] = opp_f;
                }
                if f21 == opp_f {
                    new_f_2.adj_f_i[oe_i_2 as usize] = oe_i_2 as i8;
                } else if f21 == face_index {
                    new_f_2.adj_f_i[oe_i_2 as usize] = oe_i_2 as i8;
                    new_f_2.adj_f[oe_i_2] = opp_f;
                }

                faces[face_index as usize] = new_f_1;
                faces[opp_f as usize] = new_f_2;

                if i11 >= 0 && f11 != face_index && f11 != opp_f {
                    let edge = if f11 < face_index {
                        (f11, i11 as u8)
                    } else {
                        (face_index, e_i_2)
                    };
                    if !edges_marked[3 * edge.0 as usize + edge.1 as usize] {
                        edges_marked[3 * edge.0 as usize + edge.1 as usize] = true;
                        edges_queue.push_back(edge);
                    }
                }
                if i22 >= 0 && f22 != face_index && f22 != opp_f {
                    faces[f22 as usize].adj_f[i22 as usize] = face_index;
                    faces[f22 as usize].adj_f_i[i22 as usize] = e_i_1 as i8;
                    let edge = if f22 < face_index {
                        (f22, i22 as u8)
                    } else {
                        (face_index, e_i_1)
                    };
                    if !edges_marked[3 * edge.0 as usize + edge.1 as usize] {
                        edges_marked[3 * edge.0 as usize + edge.1 as usize] = true;
                        edges_queue.push_back(edge);
                    }
                }
                if i12 >= 0 && f12 != face_index && f12 != opp_f {
                    faces[f12 as usize].adj_f[i12 as usize] = opp_f;
                    faces[f12 as usize].adj_f_i[i12 as usize] = oe_i_1 as i8;
                    let edge = if f12 < opp_f {
                        (f12, i12 as u8)
                    } else {
                        (opp_f, oe_i_1 as u8)
                    };
                    if !edges_marked[3 * edge.0 as usize + edge.1 as usize] {
                        edges_marked[3 * edge.0 as usize + edge.1 as usize] = true;
                        edges_queue.push_back(edge);
                    }
                }
                if i21 >= 0 && f21 != face_index && f21 != opp_f {
                    let edge = if f21 < opp_f {
                        (f21, i21 as u8)
                    } else {
                        (opp_f, oe_i_2 as u8)
                    };
                    if !edges_marked[3 * edge.0 as usize + edge.1 as usize] {
                        edges_marked[3 * edge.0 as usize + edge.1 as usize] = true;
                        edges_queue.push_back(edge);
                    }
                }
            }
        }
    }

    faces
        .into_par_iter()
        .zip(lengths.par_iter_mut())
        .zip(f.par_iter_mut())
        .zip(e.par_iter_mut())
        .for_each(|(((face, l), f), e)| {
            *l = face.l;
            f[0] = face.v[0];
            f[1] = face.v[1];
            f[2] = face.v[2];
            e.0 = face.adj_f;
            e.1 = face.adj_f_i;
        });

    (lengths, f, e)
}

fn build_cotan_laplacian_intrinsic(
    lengths: &[[f32; 3]],
    f: &[[u32; 3]],
    e: &[([u32; 3], [i8; 3])],
    nv: usize,
) -> SparseColMat<u32, f32> {
    let mut cots = Vec::new();
    lengths
        .par_iter()
        .map(|l| {
            let l1 = l[1];
            let l2 = l[2];
            let l3 = l[0];
            let l12 = l1 * l1;
            let l22 = l2 * l2;
            let l32 = l3 * l3;
            let s = (l1 + l2 + l3) / 2.;
            let a = f32::sqrt(s * (s - l1) * (s - l2) * (s - l3));
            let cotan1 = (l22 + l32 - l12) / (8. * a);
            let cotan2 = (l12 + l32 - l22) / (8. * a);
            let cotan3 = (l22 + l12 - l32) / (8. * a);
            [cotan1, cotan2, cotan3]
        })
        .collect_into_vec(&mut cots);

    let mut sum_cots = Vec::new();
    cots.par_iter()
        .zip(e.into_par_iter())
        .enumerate()
        .map(|(i, (cot, e))| {
            let mut cotan0 = cot[0];
            let mut cotan1 = cot[1];
            let mut cotan2 = cot[2];
            if e.1[1] >= 0 && e.0[1] as usize > i {
                cotan0 += cots[e.0[1] as usize][(e.1[1] as usize + 2) % 3];
            }
            if e.1[2] >= 0 && e.0[2] as usize > i {
                cotan1 += cots[e.0[2] as usize][(e.1[2] as usize + 2) % 3];
            }
            if e.1[0] >= 0 && e.0[0] as usize > i {
                cotan2 += cots[e.0[0] as usize][(e.1[0] as usize + 2) % 3];
            }
            [cotan0, cotan1, cotan2]
        })
        .collect_into_vec(&mut sum_cots);

    let mut deg = vec![0_u8; nv + 1];
    let mut faces_offsets = vec![0_u8; 6 * f.len()];
    for (i, ((face, edge), off)) in f
        .iter()
        .zip(e)
        .zip(faces_offsets.chunks_exact_mut(6))
        .enumerate()
    {
        if edge.1[0] < 0 || edge.0[0] as usize > i {
            deg[face[0] as usize] += 1;
            deg[face[1] as usize] += 1;
            off[0] = deg[face[0] as usize];
            off[1] = deg[face[1] as usize];
        }
        if edge.1[1] < 0 || edge.0[1] as usize > i {
            deg[face[1] as usize] += 1;
            deg[face[2] as usize] += 1;
            off[2] = deg[face[1] as usize];
            off[3] = deg[face[2] as usize];
        }
        if edge.1[2] < 0 || edge.0[2] as usize > i {
            deg[face[2] as usize] += 1;
            deg[face[0] as usize] += 1;
            off[4] = deg[face[2] as usize];
            off[5] = deg[face[0] as usize];
        }
    }

    let mut offset = 0_usize;
    let mut offsets: Vec<u32> = deg
        .iter()
        .map(|deg| {
            // Add one to account for self
            offset += *deg as usize + 1;
            // Works since last value is 0
            (offset - (*deg as usize + 1)) as u32
        })
        .collect();

    assert!(deg[nv] == 0);

    let mut indices: Vec<u32> = vec![0; offset];
    let mut values: Vec<_> = vec![0_f32; offset];
    let atomic_indices = u32::slice_to_atomic_slice(&mut indices);
    let atomic_values = atomic_from_mut_slice_f32(&mut values);

    f.par_iter()
        .zip(sum_cots.into_par_iter())
        .zip(faces_offsets.par_chunks_exact(6))
        .for_each(|((face, cots), off)| {
            let cotan0 = cots[0];
            let cotan1 = cots[1];
            let cotan2 = cots[2];
            if off[0] > 0 {
                let idx1 = offsets[face[0] as usize] as usize + off[0] as usize;
                let idx2 = offsets[face[1] as usize] as usize + off[1] as usize;
                face[1].write_to_atomic(&atomic_indices[idx1]);
                atomic_values[idx1].store((-cotan2).to_bits(), Ordering::Relaxed);

                face[0].write_to_atomic(&atomic_indices[idx2]);
                atomic_values[idx2].store((-cotan2).to_bits(), Ordering::Relaxed);
            }
            if off[2] > 0 {
                let idx1 = offsets[face[1] as usize] as usize + off[2] as usize;
                let idx2 = offsets[face[2] as usize] as usize + off[3] as usize;
                face[2].write_to_atomic(&atomic_indices[idx1]);
                atomic_values[idx1].store((-cotan0).to_bits(), Ordering::Relaxed);

                face[1].write_to_atomic(&atomic_indices[idx2]);
                atomic_values[idx2].store((-cotan0).to_bits(), Ordering::Relaxed);
            }
            if off[4] > 0 {
                let idx1 = offsets[face[2] as usize] as usize + off[4] as usize;
                let idx2 = offsets[face[0] as usize] as usize + off[5] as usize;
                face[0].write_to_atomic(&atomic_indices[idx1]);
                atomic_values[idx1].store((-cotan1).to_bits(), Ordering::Relaxed);

                face[2].write_to_atomic(&atomic_indices[idx2]);
                atomic_values[idx2].store((-cotan1).to_bits(), Ordering::Relaxed);
            }
        });

    offsets[..(offsets.len() - 1)]
        .par_iter()
        .enumerate()
        .for_each(|(i, off)| {
            (i as u32).write_to_atomic(&atomic_indices[*off as usize]);
        });

    fn insertion_sort<T: Copy + PartialOrd, U: Copy>(data_1: &mut [T], data_2: &mut [U]) {
        for i in 1..data_1.len() {
            let aux = data_1[i];
            let aux_2 = data_2[i];
            let mut j = i;
            while j > 0 && data_1[j - 1] > aux {
                data_1[j] = data_1[j - 1];
                data_2[j] = data_2[j - 1];
                j -= 1;
            }
            data_1[j] = aux;
            data_2[j] = aux_2;
        }
    }

    let duplicates = Mutex::new(Vec::new());
    rayon::iter::split(
        SubSlices2 {
            idx: &offsets,
            data_1: &mut indices,
            data_2: &mut values,
        },
        SubSlices2::splitter,
    )
    .for_each(|s| {
        let r = s.idx[0];
        for offs in s.idx.windows(2) {
            let slice_i = &mut s.data_1[(offs[0] - r) as usize..(offs[1] - r) as usize];
            let slice_v = &mut s.data_2[(offs[0] - r) as usize..(offs[1] - r) as usize];
            let vertex = slice_i[0];
            slice_v[0] = -slice_v[1..].iter().fold(0., |acc, x| acc + x);
            insertion_sort(slice_i, slice_v);
            //slice.sort_unstable_by_key(|item| item.0);
            for (i, win) in slice_i.windows(2).enumerate() {
                if win[0] == win[1] {
                    duplicates
                        .lock()
                        .unwrap()
                        .push((offs[0] + i as u32, vertex));
                }
            }
        }
    });

    if let Ok(mut duplicates) = duplicates.lock() {
        if !duplicates.is_empty() {
            duplicates.sort_unstable_by_key(|item| item.0);
            let mut offset = 0;
            'outer: for i in 0..indices.len() {
                let normal = (i + offset) as u32 != duplicates[offset].0;
                while (i + offset) as u32 == duplicates[offset].0 {
                    values[i + offset] += values[i + offset + 1];
                    values[i + offset + 1] = values[i + offset];
                    indices[i] = indices[i + offset];
                    values[i] = values[i + offset];
                    offset += 1;
                    if offset == duplicates.len() {
                        for j in (i + 1)..(indices.len() - offset) {
                            indices[j] = indices[j + offset];
                            values[j] = values[j + offset];
                        }
                        break 'outer;
                    }
                }
                if normal {
                    indices[i] = indices[i + offset];
                    values[i] = values[i + offset];
                }
            }
            indices.truncate(indices.len() - duplicates.len());
            values.truncate(values.len() - duplicates.len());

            duplicates.sort_unstable_by_key(|item| item.1);
            let mut offset = 0;
            for i in 0..offsets.len() {
                offsets[i] -= offset;
                while (offset as usize) < duplicates.len()
                    && i == duplicates[offset as usize].1 as usize
                {
                    offset += 1;
                }
            }
        }
    }

    let symbolic = SymbolicSparseColMat::new_checked(nv, nv, offsets, None, indices);
    SparseColMat::<u32, f32>::new(symbolic, values)
}

fn build_edge_map(f: &[[u32; 3]], nv: usize) -> Vec<([u32; 3], [i8; 3])> {
    let mut faces_deg = vec![0_u8; nv + 1];
    let mut faces_offsets = vec![0_u8; 3 * f.len()];
    for (row, off) in f.iter().zip(faces_offsets.chunks_exact_mut(3)) {
        if row[0] < row[1] {
            off[0] = faces_deg[row[0] as usize] as u8;
            faces_deg[row[0] as usize] += 1;
        } else {
            off[0] = faces_deg[row[1] as usize] as u8;
            faces_deg[row[1] as usize] += 1;
        }
        if row[1] < row[2] {
            off[1] = faces_deg[row[1] as usize] as u8;
            faces_deg[row[1] as usize] += 1;
        } else {
            off[1] = faces_deg[row[2] as usize] as u8;
            faces_deg[row[2] as usize] += 1;
        }
        if row[0] < row[2] {
            off[2] = faces_deg[row[0] as usize] as u8;
            faces_deg[row[0] as usize] += 1;
        } else {
            off[2] = faces_deg[row[2] as usize] as u8;
            faces_deg[row[2] as usize] += 1;
        }
    }

    let mut offset = 0;
    let faces_deg: Vec<_> = faces_deg
        .into_iter()
        .map(|v| {
            let value = v as u32;
            offset += value;
            offset - value
        })
        .collect();

    let edges_to_faces: Vec<_> = std::iter::repeat_with(|| (AtomicU32::new(0), AtomicU32::new(0)))
        .take(offset as usize)
        .collect();
    let cur_offset = faces_deg;

    f.par_iter()
        .enumerate()
        .zip(faces_offsets.par_chunks_exact(3))
        .for_each(|((i, row), f_off)| {
            if row[0] < row[1] {
                edges_to_faces[cur_offset[row[0] as usize] as usize + f_off[0] as usize]
                    .0
                    .store((row[1] << 1) as u32, Ordering::Relaxed);
                edges_to_faces[cur_offset[row[0] as usize] as usize + f_off[0] as usize]
                    .1
                    .store((i << 1) as u32, Ordering::Relaxed);
            } else {
                edges_to_faces[cur_offset[row[1] as usize] as usize + f_off[0] as usize]
                    .0
                    .store((row[0] << 1) as u32, Ordering::Relaxed);
                edges_to_faces[cur_offset[row[1] as usize] as usize + f_off[0] as usize]
                    .1
                    .store((i << 1) as u32, Ordering::Relaxed);
            }
            if row[1] < row[2] {
                edges_to_faces[cur_offset[row[1] as usize] as usize + f_off[1] as usize]
                    .0
                    .store((row[2] << 1) as u32, Ordering::Relaxed);
                edges_to_faces[cur_offset[row[1] as usize] as usize + f_off[1] as usize]
                    .1
                    .store(((i << 1) + 1) as u32, Ordering::Relaxed);
            } else {
                edges_to_faces[cur_offset[row[2] as usize] as usize + f_off[1] as usize]
                    .0
                    .store((row[1] << 1) as u32, Ordering::Relaxed);
                edges_to_faces[cur_offset[row[2] as usize] as usize + f_off[1] as usize]
                    .1
                    .store(((i << 1) + 1) as u32, Ordering::Relaxed);
            }
            if row[0] < row[2] {
                edges_to_faces[cur_offset[row[0] as usize] as usize + f_off[2] as usize]
                    .0
                    .store(((row[2] << 1) + 1) as u32, Ordering::Relaxed);
                edges_to_faces[cur_offset[row[0] as usize] as usize + f_off[2] as usize]
                    .1
                    .store((i << 1) as u32, Ordering::Relaxed);
            } else {
                edges_to_faces[cur_offset[row[2] as usize] as usize + f_off[2] as usize]
                    .0
                    .store(((row[0] << 1) + 1) as u32, Ordering::Relaxed);
                edges_to_faces[cur_offset[row[2] as usize] as usize + f_off[2] as usize]
                    .1
                    .store((i << 1) as u32, Ordering::Relaxed);
            }
        });
    let edges: Vec<_> = vec![([0; 3], [-1; 3]); f.len()]
        .into_iter()
        .map(|(v1, v2)| {
            (
                [
                    AtomicU32::from(v1[0]),
                    AtomicU32::from(v1[1]),
                    AtomicU32::from(v1[2]),
                ],
                [
                    AtomicI8::from(v2[0]),
                    AtomicI8::from(v2[1]),
                    AtomicI8::from(v2[2]),
                ],
            )
        })
        .collect();

    let mut edges_to_faces: Vec<_> = edges_to_faces
        .into_iter()
        .map(|(e1, e2)| (e1.into_inner(), e2.into_inner()))
        .collect();

    rayon::iter::split(
        SubSlices {
            idx: &cur_offset,
            data: &mut edges_to_faces,
        },
        SubSlices::splitter,
    )
    .for_each(|s| {
        let r = s.idx[0];
        for offs in s.idx.windows(2) {
            let slice = &mut s.data[(offs[0] - r) as usize..(offs[1] - r) as usize];
            slice.sort_unstable_by_key(|item| item.0);
            let mut i = 0;
            while slice.len() > 0 && i < (slice.len() - 1) {
                let item1 = slice[i];
                let item2 = slice[i + 1];
                if item1.0 >> 1 == item2.0 >> 1 {
                    let f1 = item1.1 >> 1;
                    let f2 = item2.1 >> 1;
                    let e_i_1 = 2 * (item1.0 & 1) + (item1.1 & 1);
                    let e_i_2 = 2 * (item2.0 & 1) + (item2.1 & 1);
                    edges[f1 as usize].0[e_i_1 as usize].store(f2, Ordering::Relaxed);
                    edges[f1 as usize].1[e_i_1 as usize].store(e_i_2 as i8, Ordering::Relaxed);
                    edges[f2 as usize].0[e_i_2 as usize].store(f1, Ordering::Relaxed);
                    edges[f2 as usize].1[e_i_2 as usize].store(e_i_1 as i8, Ordering::Relaxed);
                    i += 2;
                } else {
                    i += 1;
                }
            }
        }
    });
    edges
        .into_iter()
        .map(|(e, e_i)| {
            let [e0, e1, e2] = e;
            let [e_i0, e_i1, e_i2] = e_i;
            (
                [e0.into_inner(), e1.into_inner(), e2.into_inner()],
                [e_i0.into_inner(), e_i1.into_inner(), e_i2.into_inner()],
            )
        })
        .collect()
}

struct SubSlices<'a, 'b, T> {
    idx: &'a [u32],
    data: &'b mut [T],
}

impl<'a, 'b, T> SubSlices<'a, 'b, T> {
    fn splitter(self) -> (Self, Option<Self>) {
        if self.idx.len() <= 2 {
            return (self, None);
        }
        let mid = self.idx.len() / 2;
        let (idx_r, idx_l) = (&self.idx[0..mid + 1], &self.idx[mid..]);
        let (data_r, data_l) = self
            .data
            .split_at_mut(idx_l[0] as usize - idx_r[0] as usize);
        (
            Self {
                idx: idx_r,
                data: data_r,
            },
            Some(Self {
                idx: idx_l,
                data: data_l,
            }),
        )
    }
}

struct SubSlices2<'a, 'b, 'c, T, U> {
    idx: &'a [u32],
    data_1: &'b mut [T],
    data_2: &'c mut [U],
}

impl<'a, 'b, 'c, T, U> SubSlices2<'a, 'b, 'c, T, U> {
    fn splitter(self) -> (Self, Option<Self>) {
        if self.idx.len() <= 2 {
            return (self, None);
        }
        let mid = self.idx.len() / 2;
        let (idx_r, idx_l) = (&self.idx[0..mid + 1], &self.idx[mid..]);
        let (data_1_r, data_1_l) = self
            .data_1
            .split_at_mut(idx_l[0] as usize - idx_r[0] as usize);
        let (data_2_r, data_2_l) = self
            .data_2
            .split_at_mut(idx_l[0] as usize - idx_r[0] as usize);
        (
            Self {
                idx: idx_r,
                data_1: data_1_r,
                data_2: data_2_r,
            },
            Some(Self {
                idx: idx_l,
                data_1: data_1_l,
                data_2: data_2_l,
            }),
        )
    }
}
