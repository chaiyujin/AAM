#pragma once

#include <core/aam.h>

namespace aam {
	class PCA {
	private:
		int numComponents;
		Eigen::MatrixXd W;
		RowVectorX mean;
		RowVectorX eigenValues;
	public:
		PCA(const Eigen::MatrixXd &samplesOnRows, double percentage);
		template <class T>
		T project(const T &matrix) {
			auto m = matrix.rowwise() - mean;
			return m * W;
		}
		template <class T>
		T projectBack(const T &projected) {
			auto m = projected * W.transpose();
			return m.rowwise() + mean;
		}
		int getNumComponents() const { return numComponents; }
		const RowVectorX &getEigenValues() const {
			return eigenValues;
		}
	};
}