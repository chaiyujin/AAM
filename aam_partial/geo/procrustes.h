#pragma once
#include <core/aam.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace aam {
	class Procrustes {
	private:
		static RowVectorX m_meanShape;
	public:
		static Scalar procrustes(Eigen::Ref<const RowVectorX> X, Eigen::Ref<RowVectorX> Y);
		static MatrixX alignShapes(Eigen::Ref<const MatrixX> X, int maxIterations);
		static const RowVectorX &getMeanShape() {
			if (!(m_meanShape.rows() > 0)) {
				fprintf(stderr, "call 'generalizedProcrustes()' before 'getMeanShape()'. ");
				exit(-1);
			}
			return m_meanShape;
		}
	};
}