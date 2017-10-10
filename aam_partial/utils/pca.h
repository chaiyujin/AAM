#pragma once

#include <core/aam.h>
#include <utils/string.h>

namespace aam {
	class PCA {
	private:
		int numComponents;
		MatrixX W;
		RowVectorX mean;
		RowVectorX eigenValues;
	public:
		PCA(): mean(1, 1), eigenValues(0, 0), W(0, 0) {}
		PCA(const MatrixX &samplesOnRows, Scalar percentage) {
			// center aligned
			mean = samplesOnRows.colwise().mean();
			auto aligned = samplesOnRows.rowwise() - mean;
			Eigen::JacobiSVD<MatrixX> svd(aligned, Eigen::ComputeThinV);
			eigenValues = svd.singularValues();
			Scalar *sum = new Scalar[eigenValues.size()];
			sum[0] = eigenValues[0];
			for (int i = 1; i < eigenValues.size(); ++i) {
				sum[i] = sum[i - 1] + eigenValues[i];
			}
			numComponents = eigenValues.size();
			for (int i = 0; i < eigenValues.size(); ++i) {
				if (sum[i] / sum[eigenValues.size() - 1] >= percentage) {
					numComponents = i + 1;
					break;
				}
			}
			delete sum;
			W = svd.matrixV().leftCols(numComponents);
		}
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
		std::string toString() {
			std::string ret = "";
			ret += String::numToBytes<int>(numComponents);
			ret += String::matToBytes<MatrixX, Scalar>(W);
			ret += String::matToBytes<RowVectorX, Scalar>(mean);
			ret += String::matToBytes<RowVectorX, Scalar>(eigenValues);
			return ret;
		}
		void fromistream(std::istream &str) {
			String::numFromBytes<int>(str, numComponents);
			String::matFromBytes<MatrixX, Scalar>(str, W);
			String::matFromBytes<RowVectorX, Scalar>(str, mean);
			String::matFromBytes<RowVectorX, Scalar>(str, eigenValues);
		}
	};
}
