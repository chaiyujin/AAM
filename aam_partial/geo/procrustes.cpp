#include "procrustes.h"

namespace aam {
	RowVectorX Procrustes::m_meanShape;

	Scalar Procrustes::procrustes(Eigen::Ref<const RowVectorX> X_, Eigen::Ref<RowVectorX> Y_) {
		auto X = toSeparatedViewConst<Scalar>(X_);
		auto Y = toSeparatedView<Scalar>(Y_);

		RowVector2 meanX = X.colwise().mean();
		RowVector2 meanY = Y.colwise().mean();

		MatrixX centeredX = X.rowwise() - meanX;
		MatrixX centeredY = Y.rowwise() - meanY;

		// Compute Frobenius norm. 
		const Scalar sX = centeredX.norm();
		const Scalar sY = centeredY.norm();

		// Scale to unit norm
		centeredX /= sX;
		centeredY /= sY;

		// Find optimal rotation based on correlation of landmarks
		MatrixX A = centeredX.transpose() * centeredY;
		auto svd = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);

		// Equation 7.
		Matrix2 v = svd.matrixV();
		Matrix2 u = svd.matrixU();
		RowVector2 s = svd.singularValues();
		Matrix2 rot = v * u.transpose();

		// Make sure we don't suffer from reflection.
		if (rot.determinant() < 0) {
			v.rightCols(1) *= -1;
			s.rightCols(1) *= -1;
			rot = v * u.transpose();
		}

		Scalar trace = s.sum();

		// Scaling of Y
		// Scalar b = trace * sX / sY;

		// Distance of X and T(Y)
		Scalar d = 1 - trace * trace;

		// Transform Y
		Y = (((centeredY * rot) * trace * sX).rowwise() + meanX).eval();

		return d;
	}

	MatrixX Procrustes::alignShapes(Eigen::Ref<const MatrixX> X, int maxIterations)
	{
		const MatrixX::Index nShapes = X.rows();

		MatrixX alignedShapes = X;

		// Perform iterative optimization
		// - arbitrarily choose a reference shape(typically by selecting it among the available instances)
		// - superimpose all instances to current reference shape
		// - compute the mean shape of the current set of superimposed shapes
		// - if the Procrustes distance between mean and reference shape is above a threshold, set reference to mean shape and continue to step 2.

		bool done = false;
		RowVectorX refShape = alignedShapes.row(0);
		Scalar lastDist = std::numeric_limits<Scalar>::max();
		int iterations = 0;
		do {
			for (MatrixX::Index s = 0; s < nShapes; ++s) {
				procrustes(refShape, alignedShapes.row(s));
			}

			m_meanShape = RowVectorX::Zero(X.cols());
			for (MatrixX::Index s = 0; s < nShapes; ++s) {
				m_meanShape += alignedShapes.row(s);
			}
			m_meanShape /= (Scalar)nShapes;

			Scalar dist = (m_meanShape - refShape).norm();
			if (dist > lastDist || ++iterations > maxIterations)
				done = true;


			lastDist = dist;
			refShape = m_meanShape;

		} while (!done);

		for (MatrixX::Index s = 0; s < nShapes; ++s) {
			procrustes(refShape, alignedShapes.row(s));
		}

		return alignedShapes;
	}
}