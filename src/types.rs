#[derive(Clone)]
pub enum SurfaceIndices {
    Triangles(Vec<[u32; 3]>),
    Quads(Vec<[u32; 4]>),
    Polygons(Vec<u32>, Vec<u32>),
}

impl SurfaceIndices {
    pub fn size(&self) -> usize {
        match self {
            SurfaceIndices::Triangles(t) => t.len(),
            SurfaceIndices::Quads(q) => q.len(),
            SurfaceIndices::Polygons(_i, s) => s.len(),
        }
    }

    pub fn tot_triangles(&self) -> usize {
        match self {
            SurfaceIndices::Triangles(t) => t.len(),
            SurfaceIndices::Quads(q) => 2 * q.len(),
            SurfaceIndices::Polygons(_i, s) => {
                s.into_iter().fold(0, |acc, size| acc + *size as usize - 2)
            }
        }
    }

    pub fn len(&self) -> usize {
        self.size()
    }
}

impl Into<SurfaceIndices> for Vec<[u32; 3]> {
    fn into(self) -> SurfaceIndices {
        SurfaceIndices::Triangles(self)
    }
}

impl Into<SurfaceIndices> for Vec<[u32; 4]> {
    fn into(self) -> SurfaceIndices {
        SurfaceIndices::Quads(self)
    }
}

impl Into<SurfaceIndices> for (Vec<u32>, Vec<u32>) {
    fn into(self) -> SurfaceIndices {
        SurfaceIndices::Polygons(self.0, self.1)
    }
}

impl Into<SurfaceIndices> for (Vec<u32>, Vec<u8>) {
    fn into(self) -> SurfaceIndices {
        let strides_32 = self.1.into_iter().map(|s| s as _).collect();
        SurfaceIndices::Polygons(self.0, strides_32)
    }
}

//impl Into<SurfaceIndices> for Vec<Vec<u32>> {
//
//}

pub struct SurfaceIndicesIntoIterator<'a> {
    indices: &'a SurfaceIndices,
    index: usize,
    stride_index: usize,
}

impl<'a> IntoIterator for &'a SurfaceIndices {
    type Item = &'a [u32];
    type IntoIter = SurfaceIndicesIntoIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        SurfaceIndicesIntoIterator {
            indices: self,
            index: 0,
            stride_index: 0,
        }
    }
}

use std::borrow::Borrow;

impl<'a> Iterator for SurfaceIndicesIntoIterator<'a> {
    type Item = &'a [u32];

    fn next(&mut self) -> Option<&'a [u32]> {
        let res = match self.indices {
            SurfaceIndices::Triangles(t) => t.get(self.index).map(|a| a.borrow()),
            SurfaceIndices::Quads(q) => q.get(self.index).map(|a| a.borrow()),
            SurfaceIndices::Polygons(i, s) => s.get(self.index).map(|size| {
                let res = &i[self.stride_index..self.stride_index + *size as usize];
                self.stride_index += *size as usize;
                res
            }),
        };
        self.index += 1;
        res
    }
}

pub trait Vertices {
    fn into(self) -> Vec<[f32; 3]>;
}

impl Vertices for Vec<[f32; 3]> {
    fn into(self) -> Vec<[f32; 3]> {
        self
    }
}

impl Vertices for ndarray::Array2<f32> {
    fn into(self) -> Vec<[f32; 3]> {
        self.rows()
            .into_iter()
            .map(|row| [row[0], row[1], row[2]])
            .collect()
    }
}

impl Vertices for ndarray::Array2<f64> {
    fn into(self) -> Vec<[f32; 3]> {
        self.rows()
            .into_iter()
            .map(|row| [row[0] as f32, row[1] as f32, row[2] as f32])
            .collect()
    }
}

impl Vertices for nalgebra::base::MatrixXx3<f32> {
    fn into(self) -> Vec<[f32; 3]> {
        self.row_iter()
            .map(|row| [row[0], row[1], row[2]])
            .collect()
    }
}

pub trait Vertices2D {
    fn into(self) -> Vec<[f32; 2]>;
}

impl Vertices2D for Vec<[f32; 2]> {
    fn into(self) -> Vec<[f32; 2]> {
        self
    }
}

impl Vertices2D for ndarray::Array2<f32> {
    fn into(self) -> Vec<[f32; 2]> {
        self.rows()
            .into_iter()
            .map(|row| [row[0], row[1]])
            .collect()
    }
}

impl Vertices2D for nalgebra::base::MatrixXx3<f32> {
    fn into(self) -> Vec<[f32; 2]> {
        self.row_iter().map(|row| [row[0], row[1]]).collect()
    }
}

pub trait Scalar {
    fn into(self) -> Vec<f32>;
}

impl Scalar for Vec<f32> {
    fn into(self) -> Vec<f32> {
        self
    }
}

impl Scalar for ndarray::Array1<f32> {
    fn into(self) -> Vec<f32> {
        self.into_raw_vec()
    }
}

impl Scalar for ndarray::Array1<f64> {
    fn into(self) -> Vec<f32> {
        self.into_raw_vec().into_iter().map(|f| f as f32).collect()
    }
}

impl Scalar for nalgebra::base::DVector<f32> {
    fn into(self) -> Vec<f32> {
        self.row_iter().map(|row| row[0]).collect()
    }
}

pub trait Color {
    fn into(self) -> Vec<[f32; 3]>;
}

impl Color for Vec<[f32; 3]> {
    fn into(self) -> Vec<[f32; 3]> {
        self
    }
}

impl Color for ndarray::Array2<f32> {
    fn into(self) -> Vec<[f32; 3]> {
        self.rows()
            .into_iter()
            .map(|row| [row[0], row[1], row[2]])
            .collect()
    }
}

impl Color for nalgebra::base::MatrixXx3<f32> {
    fn into(self) -> Vec<[f32; 3]> {
        self.row_iter()
            .map(|row| [row[0], row[1], row[2]])
            .collect()
    }
}
