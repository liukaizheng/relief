#include "eigen_alias.h"
#include <deform_surface.h>
#include <Eigen/svd>
#include <Eigen/Sparse>
#include <fstream>
#include <iostream>
#include <fmt/format.h>

auto write_uv(const std::string& name, const VMat& uv, const FMat& F) {
    std::ofstream file(name);
    for (Eigen::Index i = 0; i < uv.rows(); ++i) {
        file << "v " << uv(i, 0) << " " << uv(i, 1) <<  " " << uv(i, 2) << "\n";
    }
    for (Eigen::Index i = 0; i < F.rows(); ++i) {
        file << "f " << F(i, 0) + 1 << " " << F(i, 1) + 1 << " " << F(i, 2) + 1 << "\n";
    }
    file.close();
}

void map_boundary_to_circle(VMat& V, const Eigen::Index n_boundary) {
    auto V1 = V.topRows(n_boundary);
    using IVec = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;
    IVec I = IVec::LinSpaced(n_boundary, 1, n_boundary);
    I[n_boundary - 1] = 0;
    auto L = (V1 - V1(I, Eigen::placeholders::all)).rowwise().norm().eval();
    const auto l = L.sum();
    const auto r = l / (2 * M_PI);
    L /= l;
    double theta = 0;
    for (Eigen::Index i = 0; i < n_boundary; ++i) {
        const auto temp = L(i);
        L(i) = theta;
        theta += temp;
    }
    L *= 2 * M_PI;
    V1.col(0) = L.array().cos() * r;
    V1.col(1) = L.array().sin() * r;
    V1.col(2).setZero();

    std::cout << "V1" << V1 << std::endl;
}

template<typename DerivedV1, typename DerivedV2, typename DerivedN>
void cross(const Eigen::MatrixBase<DerivedV1>& V1, const Eigen::MatrixBase<DerivedV2>& V2, Eigen::MatrixBase<DerivedN>& N) {
    N.col(0) = V1.col(1).cwiseProduct(V2.col(2)) - V1.col(2).cwiseProduct(V2.col(1));
    N.col(1) = V1.col(2).cwiseProduct(V2.col(0)) - V1.col(0).cwiseProduct(V2.col(2));
    N.col(2) = V1.col(0).cwiseProduct(V2.col(1)) - V1.col(1).cwiseProduct(V2.col(0));
}

template<typename DerivedV, typename DerivedF>
auto tri_gradients(const Eigen::MatrixBase<DerivedV>& vertices, const Eigen::MatrixBase<DerivedF>& faces) noexcept {
    const auto face_count = faces.rows(); // number of triangles
    const auto vertex_count = vertices.rows(); // number of vertices
    auto edge_bc = (vertices(faces.col(2), Eigen::placeholders::all) - vertices(faces.col(1), Eigen::placeholders::all)).eval(); // edge B->C
    auto edge_ca = (vertices(faces.col(0), Eigen::placeholders::all) - vertices(faces.col(2), Eigen::placeholders::all)).eval(); // edge C->A
    using FaceVectors = Eigen::Matrix<typename DerivedV::Scalar, Eigen::Dynamic, 3>; // reusable per-face 3D rows
    FaceVectors normals(face_count, 3); // face normals (scaled)
    cross(edge_bc, edge_ca, normals);
    auto double_area = normals.rowwise().norm().eval(); // per-face 2*area
    auto area_repeated = double_area.replicate(1, 3).array(); // broadcasted for division
    normals = normals.array() / area_repeated;

    FaceVectors grad_phi0(face_count, 3); // grad of barycentric phi0 (vertex 0)
    cross(normals, edge_bc, grad_phi0);
    grad_phi0 = grad_phi0.array() / area_repeated;

    FaceVectors grad_phi1(face_count, 3); // grad of barycentric phi1 (vertex 1)
    cross(normals, edge_ca, grad_phi1);
    grad_phi1 = grad_phi1.array() / area_repeated;

    Eigen::SparseMatrix<Eigen::Index> face_vertex_map(face_count, vertex_count); // maps face->vertex slot
    std::vector<Eigen::Triplet<Eigen::Index>> fv_entries; // buffer for map entries
    Eigen::Index entry = 0; // running triplet id
    for (Eigen::Index i = 0; i < face_count; i++) {
        fv_entries.emplace_back(i, faces(i, 0), entry++);
        fv_entries.emplace_back(i, faces(i, 1), entry++);
        fv_entries.emplace_back(i, faces(i, 2), entry++);
    }
    face_vertex_map.setFromTriplets(fv_entries.begin(), fv_entries.end());
    face_vertex_map.makeCompressed();

    IVec face_rows = IVec::LinSpaced(face_count, 0, face_count * 3 - 3); // row offsets per face
    FaceVectors grad_rows(face_count * 3, 3); // stacked phi gradients per face
    grad_rows(face_rows, Eigen::placeholders::all) = grad_phi0;
    grad_rows(face_rows.array() + 1, Eigen::placeholders::all) = grad_phi1;
    grad_rows(face_rows.array() + 2, Eigen::placeholders::all) = -grad_phi0 - grad_phi1;

    return std::make_tuple(std::move(edge_bc), std::move(grad_phi0), std::move(normals), std::move(double_area), std::move(face_vertex_map), std::move(grad_rows));
}

auto build_system_pattern1(const RowSpMat& DX, const Eigen::Index n_fixed) noexcept {
    const Eigen::Vector2i I1 {0, 3};
    const Eigen::Vector2i I2 {1, 4};
    const Eigen::Vector2i I3 {2, 5};
    using IVec = Eigen::Matrix<RowSpMat::StorageIndex, 1, Eigen::Dynamic, Eigen::RowMajor>;
    using Mat6 = Eigen::Matrix<RowSpMat::StorageIndex, 6, Eigen::Dynamic, Eigen::RowMajor>;
    const auto n_free = DX.innerSize() - n_fixed; // number of free vertices
    std::vector<RowSpMat::StorageIndex> pattern_outer(DX.outerSize() * 6 + 1); // row pointer for 4x block rows
    std::vector<RowSpMat::StorageIndex> pattern_inner(DX.nonZeros() * 6); // column indices for duplicated blocks
    pattern_outer[0] = 0;
    const auto dx_outer_ptr = DX.outerIndexPtr(); // source row starts
    const auto dx_inner_ptr = DX.innerIndexPtr(); // source column ids
    std::size_t outer_cursor = 0; // write head for pattern_outer
    RowSpMat::StorageIndex value_cursor = 0; // write head for pattern_inner
    std::vector<Eigen::Index> skip_counts(DX.outerSize()); // leading constrained entries to skip per row
    for (Eigen::Index i = 0; i < DX.outerSize(); i++) {
        auto dx_start = dx_outer_ptr[i];
        const auto dx_end = dx_outer_ptr[i + 1];
        const auto row_inner_ptr = dx_inner_ptr + dx_start; // pointer to this row's inner ids
        skip_counts[i] = std::find_if(row_inner_ptr, dx_inner_ptr + dx_end, [n_fixed](auto x) noexcept { return x >=  n_fixed; }) - row_inner_ptr;
        dx_start += skip_counts[i];
        const auto row_size = dx_end - dx_start; // free entries in this row
        const auto row_inner_indices = IVec::Map(dx_inner_ptr + dx_start, row_size).array(); // free vertex ids
        auto block_indices = Mat6::Map(pattern_inner.data() + value_cursor, 6, row_size);

        block_indices(I1, Eigen::placeholders::all) = (row_inner_indices - n_fixed).replicate<2, 1>();
        block_indices(I2, Eigen::placeholders::all) = (row_inner_indices + (n_free - n_fixed)).replicate<2, 1>();
        block_indices(I3, Eigen::placeholders::all) = (row_inner_indices +(2 * n_free - n_fixed)).replicate<2, 1>();

        IVec::Map(pattern_outer.data() + outer_cursor + 1, 6) = IVec::LinSpaced(6, value_cursor + row_size, value_cursor + row_size * 6);
        value_cursor += row_size * 6;
        outer_cursor += 6;
    }
    pattern_inner.resize(value_cursor);
    RowSpMat A = RowSpMat::Map(DX.rows() * 6, n_free * 3, pattern_inner.size(), pattern_outer.data(), pattern_inner.data(), std::vector<double>(pattern_inner.size()).data());
    return std::make_pair(std::move(A), std::move(skip_counts));
}

void fill_system_values(const RowSpMat& DX, const RowSpMat& DY, const VMat4& W, const std::vector<Eigen::Index>& skip_counts, const VMat& XYZ, RowSpMat& A, Eigen::VectorXd& boundary_rhs) noexcept {
    using RowValues = Eigen::Matrix<RowSpMat::Scalar, 1, Eigen::Dynamic>; // dense row view of gradient values
    using IndexRow = Eigen::Matrix<RowSpMat::StorageIndex, 1, Eigen::Dynamic, Eigen::RowMajor>; // row of vertex ids
    using Block6 = Eigen::Matrix<RowSpMat::Scalar, 6, Eigen::Dynamic, Eigen::RowMajor>; // 4x face block of system
    const auto row_ptr = A.outerIndexPtr(); // offsets into value/inner arrays
    boundary_rhs.setZero();
    for (Eigen::Index face = 0; face < DX.rows(); face++) {
        const auto block_row = face * 6; // starting block row
        const auto block_start = row_ptr[block_row]; // value offset
        const auto block_end = row_ptr[block_row + 1];
        const auto block_size = block_end - block_start; // total cols for this face (u+v)
        const auto total_grad_cols = block_size + skip_counts[face]; // includes fixed part
        const auto dx_row = RowValues::Map(DX.valuePtr() + DX.outerIndexPtr()[face], total_grad_cols); // Dx entries (fixed+free)
        const auto dy_row = RowValues::Map(DY.valuePtr() + DY.outerIndexPtr()[face], total_grad_cols); // Dy entries (fixed+free)
        const auto weight_row = W.row(face).transpose().array().replicate(1, total_grad_cols); // polar weights
        const auto w = W.row(face).reshaped(Eigen::fix<2>, Eigen::fix<2>);
        const auto dx_block = (w(0, 0) * dx_row + w(0, 1) * dy_row).eval();
        const auto dy_block = (w(1, 0) * dx_row + w(1, 1) * dy_row).eval();
        auto block_values = Block6::Map(A.valuePtr() + block_start, 6, block_size); // target block in A
        block_values.topRows(3).rowwise() = dx_block.tail(block_size);
        block_values.bottomRows(3).rowwise() = dy_block.tail(block_size);

        auto fixed_indices = IndexRow::Map(DX.innerIndexPtr() + DX.outerIndexPtr()[face], skip_counts[face]); // constrained vertex ids
        auto fixed_XYZ = XYZ(fixed_indices, Eigen::placeholders::all);

        auto rhs_slice = boundary_rhs.segment(block_row, 6); // slice for this face
        rhs_slice.segment(0, 3) += (dx_block.head(skip_counts[face]) * fixed_XYZ).transpose();
        rhs_slice.segment(3, 3) += (dy_block.head(skip_counts[face]) * fixed_XYZ).transpose();
    }
}

void assign_rhs(const VMat4& W, const VMat6& R, Eigen::VectorXd& rhs) noexcept {
    Eigen::Index idx = 0;
    for (Eigen::Index i = 0; i < W.rows(); ++i) {
        rhs.segment(idx, 6).reshaped<Eigen::RowMajor>(Eigen::fix<2>, Eigen::fix<3>) =
            W.row(i).reshaped(Eigen::fix<2>, Eigen::fix<2>)
            * R.row(i).reshaped<Eigen::RowMajor>(Eigen::fix<2>, Eigen::fix<3>);
        idx += 6;
    }
}

DeformSurface::DeformSurface(VMat&& vertices, FMat&& faces, const Eigen::Index n_fixed_vertices) : vertices(vertices), faces(faces), n_fixed_vertices(n_fixed_vertices) {}

void DeformSurface::deform(const std::size_t n_iterations, VMat& new_vertices) {
    // new_vertices = this->vertices;
    // new_vertices.topRows(this->n_fixed_vertices).array() -= 0.1;
    // map_boundary_to_circle(new_vertices, this->n_fixed_vertices);
    write_uv("output0.obj", new_vertices, faces);
    using Mat23 = Eigen::Matrix<double, 2, 3>;
    Eigen::Index n_free = this->vertices.rows() - n_fixed_vertices;
    auto [B1, B2, N, DA, IFV, G] = tri_gradients(this->vertices, this->faces);
    DA /= DA.sum();

    // Compute per-face gradient dot-products for a barycentric basis vector B and
    // write the scalar results into a dense buffer matching the IFV/G ordering.
    auto assemble_gradient_values = [&](const auto& B, double* values) noexcept {
        const auto outer_indices = IFV.outerIndexPtr();
        const auto inner_indices = IFV.innerIndexPtr();
        for (Eigen::Index i = 0; i < IFV.outerSize(); i++) {
            const auto start = outer_indices[i];
            const auto end = outer_indices[i + 1];
            const auto size = end - start;
            const auto I1 = Eigen::Matrix<decltype(IFV)::Scalar, Eigen::Dynamic, 1>::Map(IFV.valuePtr() + start, size);
            const auto I2 = Eigen::Matrix<SpMat::StorageIndex, Eigen::Dynamic, 1>::Map(inner_indices + start, size);
            Eigen::VectorXd::Map(values + start, size) = (G(I1, Eigen::placeholders::all).array() * B(I2, Eigen::placeholders::all).array()).rowwise().sum();
        }
    };

    B1 = B1.array() / B1.rowwise().norm().replicate<1, 3>().array();
    B2 = B2.array() / B2.rowwise().norm().replicate<1, 3>().array();
    RowSpMat Du;
    RowSpMat Dv;
    {
        std::vector<double> temp(IFV.nonZeros());
        assemble_gradient_values(B1, temp.data());
        Du = SpMat::Map(IFV.rows(), IFV.cols(), IFV.nonZeros(), IFV.outerIndexPtr(), IFV.innerIndexPtr(), temp.data());
        assemble_gradient_values(B2, temp.data());
        Dv = SpMat::Map(IFV.rows(), IFV.cols(), IFV.nonZeros(), IFV.outerIndexPtr(), IFV.innerIndexPtr(), temp.data());
    }
    VMat6 J(B1.rows(), 6);
    VMat6 R(B1.rows(), 6);
    VMat4 W(B1.rows(), 4);

    const auto update_jacobian = [&J, &Du, &Dv](const VMat& V) noexcept {
        J.col(0) = Du * V.col(0);
        J.col(1) = Dv * V.col(0);
        J.col(2) = Du * V.col(1);
        J.col(3) = Dv * V.col(1);
        J.col(4) = Du * V.col(2);
        J.col(5) = Dv * V.col(2);
    };

    std::vector<Eigen::Index> skip_indices;
    RowSpMat A;
    Eigen::VectorXd boundary_rhs(B1.rows() * 6);
    Eigen::VectorXd rhs(B1.rows() * 6);

    for (std::size_t _ = 0; _ < n_iterations; ++_) {
        update_jacobian(new_vertices);

        // update rotation and weight matrices
        for (Eigen::Index i = 0; i < J.rows(); ++i) {
            auto ji = J.row(i).reshaped(Eigen::fix<2>, Eigen::fix<3>);
            Eigen::JacobiSVD<decltype(ji), Eigen::ComputeFullU | Eigen::ComputeFullV> svd(ji);

            const auto si = svd.singularValues().array();
            // std::cout << si.transpose() << std::endl;

            // const auto si_arr = si.array();
            auto swi = (((1.0 + si) / si * (1.0 + si.square())).sqrt() / si).matrix().asDiagonal();
            R.row(i).reshaped<Eigen::RowMajor>(Eigen::fix<2>, Eigen::fix<3>) = svd.matrixU() * svd.matrixV().transpose().topRows(2);
            W.row(i).reshaped(Eigen::fix<2>, Eigen::fix<2>) = svd.matrixU() * swi * svd.matrixU().transpose();
        }
        if (skip_indices.empty()) {
            std::tie(A, skip_indices) = build_system_pattern1(Du, n_fixed_vertices);
        }
        fill_system_values(Du, Dv, W, skip_indices, new_vertices, A, boundary_rhs);
        assign_rhs(W, R, rhs);
        rhs -= boundary_rhs;

        RowSpMat At = A.transpose();
        RowSpMat L = At * DA.replicate<1, 6>().reshaped<Eigen::RowMajor>().asDiagonal() * A;
        Eigen::VectorXd real_rhs = At * DA.replicate<1, 6>().reshaped<Eigen::RowMajor>().asDiagonal() * rhs;
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > solver;
        new_vertices.bottomRows(this->vertices.rows() - n_fixed_vertices).reshaped() = solver.compute(L).solve(real_rhs);
        write_uv(fmt::format("output{}.obj", _), new_vertices, this->faces);
        const auto a = 2;

        // if (A_to_At_indices.size() == 0) {
        //     std::tie(At, A_to_At_indices) = transpose_sparse_matrix(A);
        // } else {
        //     RowSpMat::ScalarVector::Map(At.valuePtr(), A.nonZeros())(A_to_At_indices) = RowSpMat::ScalarVector::Map(A.valuePtr(), A.nonZeros());
        // }

        // if (L.size() == 0) {
        //     L = normal_cache.build(At);
        // }
        // normal_cache.assign(At, DDA, L);
        // multiply_normal_rhs(At, DDA, rhs, real_rhs);

        // Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > solver;
        // new_uv.bottomRows(V.rows() - n_bnd_points).reshaped() = solver.compute(L).solve(real_rhs);
        // energy = flip_avoiding_line_search(F, uv, new_uv, compute_energy, energy);
        // uv = new_uv;
        // std::cout << "Energy: " << energy << std::endl;

        // write_uv("uv_new.obj", new_uv, F);
    }
}
